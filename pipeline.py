import os
import json
import time
import datetime
import subprocess
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sort import SimpleTracker

# CONFIG
CAMS_FILE = "cams.json"
OUTPUT_DIR = "output"
MODEL_NAME = "yolov8n.pt"
TARGET_FPS = 2  # Process 2 frames of video per second (every 500ms)
CONF_THRESH = 0.35
WANTED = ["person", "backpack", "handbag", "bottle"]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_cams():
    with open(CAMS_FILE) as f:
        return json.load(f)

def get_video_start_time(path):
    """
    Try to get creation_time from metadata using ffprobe.
    Returns datetime object or None.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        j = json.loads(res.stdout)
        tags = j.get("format", {}).get("tags", {}) or {}
        ct = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
        if ct:
            # Handle standard ISO format from ffprobe
            try:
                return datetime.datetime.fromisoformat(ct.replace("Z", ""))
            except ValueError:
                return None
    except Exception:
        pass
    return None

def get_video_duration_ms(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return int((frame_count / fps) * 1000)
    return 0

def time_to_ms(dt):
    return int(dt.timestamp() * 1000)

def rotate_image(image, angle):
    """
    Rotate image by 90, -90, or 180 degrees.
    angle: 90 (CLOCKWISE), -90 (COUNTER_CLOCKWISE), 180
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image

def point_in_poly(x, y, poly):
    """
    Ray casting algorithm for point in polygon.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
    return inside

def main():
    ensure_dir(OUTPUT_DIR)
    cams_config = read_cams()
    
    # 1. Prepare Camera Info & Synchronization
    cam_data = {}
    print("[INFO] analyzing video files...")
    
    for name, cfg in cams_config.items():
        path = Path(cfg["file"])
        if not path.exists():
            print(f"[ERR] {name}: file not found {path}")
            continue
            
        start_dt = get_video_start_time(path)
        if not start_dt:
            # Fallback if no metadata: check config or assume 0
            if cfg.get("start_time"):
                try:
                    start_dt = datetime.datetime.fromisoformat(cfg.get("start_time"))
                except:
                    pass
        
        start_ms = time_to_ms(start_dt) if start_dt else 0
        if start_ms == 0:
            print(f"[WARN] {name}: using start time 0")
            
        duration_ms = get_video_duration_ms(path)
        
        cam_data[name] = {
            "path": str(path),
            "flip": cfg.get("flip", False),
            "rotate": cfg.get("rotate", 0),
            "zones": cfg.get("zones", {}),
            "start_ms": start_ms,
            "duration_ms": duration_ms,
            "end_ms": start_ms + duration_ms
        }
        print(f"[INFO] {name}: duration={duration_ms/1000:.1f}s")

    if not cam_data:
        print("[ERR] No valid cameras found.")
        return

    # Global time window
    valid_starts = [d["start_ms"] for d in cam_data.values()]
    valid_ends = [d["end_ms"] for d in cam_data.values()]
    global_start = min(valid_starts)
    global_end = max(valid_ends)
    
    print(f"[INFO] Processing Window: {global_start} to {global_end} (duration: {(global_end-global_start)/1000:.1f}s)")
    
    # 2. Init Model & Trackers
    print(f"[INFO] Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    trackers = {name: SimpleTracker() for name in cam_data.keys()}
    
    # 3. Init Video Capture & Writers
    caps = {}
    writers = {}
    
    for name, data in cam_data.items():
        cap = cv2.VideoCapture(data["path"])
        caps[name] = cap
        
        # Read one frame to determine output size after rotation
        ret, tmp_frame = cap.read()
        if not ret:
            print(f"[ERR] Could not read first frame of {name}")
            continue
        
        # Apply transforms to first frame to get dims
        if data["flip"]:
            tmp_frame = cv2.flip(tmp_frame, 1)
        if data["rotate"] != 0:
            tmp_frame = rotate_image(tmp_frame, data["rotate"])
            
        h, w = tmp_frame.shape[:2]
        
        outpath = os.path.join(OUTPUT_DIR, f"{name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[name] = cv2.VideoWriter(outpath, fourcc, TARGET_FPS, (w, h))
        
        # Reset capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    # 4. Processing Loop
    step_ms = int(1000 / TARGET_FPS)
    current_time = global_start
    alert_log_path = os.path.join(OUTPUT_DIR, "alerts.log")
    
    try:
        while current_time <= global_end:
            # Batch collection
            batch_frames = []
            batch_meta = [] # (name, original_frame_but_transformed)
            
            for name, data in cam_data.items():
                if name not in caps: continue
                
                # Check if camera acts in this time window
                rel_ms = current_time - data["start_ms"]
                if rel_ms < 0 or rel_ms > data["duration_ms"]:
                    continue
                    
                cap = caps[name]
                cap.set(cv2.CAP_PROP_POS_MSEC, float(rel_ms))
                ret, frame = cap.read()
                
                if ret:
                    if data["flip"]:
                        frame = cv2.flip(frame, 1)
                    if data["rotate"] != 0:
                        frame = rotate_image(frame, data["rotate"])
                    
                    batch_frames.append(frame)
                    batch_meta.append(name)
            
            if not batch_frames:
                current_time += step_ms
                continue
                
            # Inference (Batch)
            # YOLOv8 handles list of numpy arrays, but for best speed resizing to common size helps.
            # Ultralytics handles mixed sizes, but let's just pass them.
            results = model.predict(batch_frames, imgsz=640, device="cpu", conf=CONF_THRESH, verbose=False)
            
            # Post-process results
            for i, res in enumerate(results):
                name = batch_meta[i]
                frame = batch_frames[i]
                
                # Detections
                detections = []
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.model.names.get(cls_id, str(cls_id))
                    
                    if cls_name in WANTED:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append([x1, y1, x2, y2, conf, cls_name])
                
                # Tracking
                trackers[name].update(detections)
                tracks = trackers[name].get_tracks()
                
                # Annotate
                for tid, bbox, trace in tracks:
                    x1, y1, x2, y2 = map(int, bbox)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Draw Box & Label
                    color = (0, 255, 0) # Green for tracked objs
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw Trace
                    if len(trace) > 1:
                        pts = np.array(trace, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
                        
                    # Check Zones
                    for zone_name, poly in cam_data[name]["zones"].items():
                        if point_in_poly(cx, cy, poly):
                            # Alert visual
                            cv2.putText(frame, f"ALERT: {zone_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
                            
                            # Log alert
                            with open(alert_log_path, "a") as af:
                                af.write(f"{datetime.datetime.utcnow().isoformat()},{name},{tid},{zone_name}\n")
                
                # Draw Zones
                for zone_name, poly in cam_data[name]["zones"].items():
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    cv2.putText(frame, zone_name, (poly[0][0], poly[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Write to output video
                writers[name].write(frame)
                
            current_time += step_ms
            # Optional progress
            if (current_time - global_start) % 10000 < step_ms:
                prog = (current_time - global_start) / (global_end - global_start + 1e-9) * 100
                print(f"[PROG] {prog:.1f}%")
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        for c in caps.values(): c.release()
        for w in writers.values(): w.release()
        print("[DONE] Processing complete.")

if __name__ == "__main__":
    main()
