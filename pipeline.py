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

# FPS config: 10 frames per second (Fluid Surveillance)
TARGET_FPS = 10.0
STEP_MS = int(1000 / TARGET_FPS)

# Split Thresholds
CONF_THRESH_PERSON = 0.40
CONF_THRESH_OBJECT = 0.15 # Low threshold for objects
WANTED = ["person", "backpack", "handbag", "bottle", "cell phone"]

COLORS = {
    "person": (0, 255, 0),      
    "backpack": (255, 0, 0),    
    "handbag": (0, 0, 255),   
    "bottle": (0, 255, 255),    
    "cell phone": (255, 0, 255),
    "default": (200, 200, 200)
}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_cams():
    with open(CAMS_FILE) as f:
        return json.load(f)

def parse_time(t_str):
    try:
        return datetime.datetime.fromisoformat(t_str)
    except:
        return None

def get_video_duration_ms(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0: return int((cnt / fps) * 1000)
    return 0

def time_to_ms(dt):
    return int(dt.timestamp() * 1000)

def rotate_image(image, angle):
    if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    return image

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect: inside = not inside
    return inside

def get_faded_color(color, alpha=0.5):
    return tuple(int(c * alpha) for c in color)

def main():
    ensure_dir(OUTPUT_DIR)
    cams_config = read_cams()
    
    cam_data = {}
    print("[INFO] Analyzing video files (v7 - 10 FPS)...")
    
    for name, cfg in cams_config.items():
        path = Path(cfg["file"])
        if not path.exists():
            print(f"[ERR] {name}: file not found")
            continue
            
        start_dt = None
        if cfg.get("start_time"):
            start_dt = parse_time(cfg.get("start_time"))
        
        start_ms = time_to_ms(start_dt) if start_dt else 0
        dur_ms = get_video_duration_ms(path)
        
        cam_data[name] = {
            "path": str(path),
            "flip": cfg.get("flip", False),
            "rotate": cfg.get("rotate", 0),
            "zones": cfg.get("zones", {}),
            "start_dt": start_dt, 
            "start_ms": start_ms,
            "duration_ms": dur_ms,
            "end_ms": start_ms + dur_ms
        }
        print(f"[INFO] {name}: {dur_ms/1000:.1f}s")
        
    if not cam_data: return

    valid_starts = [d["start_ms"] for d in cam_data.values()]
    valid_ends = [d["end_ms"] for d in cam_data.values()]
    global_start = min(valid_starts)
    global_end = max(valid_ends)
    
    print(f"[INFO] Global Window: {global_start} - {global_end}")
    
    model = YOLO(MODEL_NAME)
    # Tracker: iou=0.3 because at 10 FPS objects are close
    # max_lost=10 = 1 second of persistence
    trackers = {name: SimpleTracker(max_lost=10, iou_threshold=0.3) for name in cam_data.keys()}
    
    caps = {}
    writers = {}
    writer_dims = {}
    
    for name, data in cam_data.items():
        cap = cv2.VideoCapture(data["path"])
        caps[name] = cap
        
        ret, tmp = cap.read()
        if not ret: continue
        
        if data["flip"]: tmp = cv2.flip(tmp, 1)
        if data["rotate"] != 0: tmp = rotate_image(tmp, data["rotate"])
            
        h, w = tmp.shape[:2]
        writer_dims[name] = (w, h)
        outpath = os.path.join(OUTPUT_DIR, f"{name}_annotated.mp4")
        writers[name] = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    current_time = global_start
    alert_log_path = os.path.join(OUTPUT_DIR, "alerts.log")
    
    try:
        while current_time <= global_end:
            batch_frames = []
            batch_meta = [] 
            
            for name, data in cam_data.items():
                if name not in writers: continue
                
                rel_ms = current_time - data["start_ms"]
                
                if rel_ms < 0 or rel_ms > data["duration_ms"]:
                    w, h = writer_dims[name]
                    # Black frame for syncing
                    writers[name].write(np.zeros((h, w, 3), dtype=np.uint8))
                    continue
                
                cap = caps[name]
                cap.set(cv2.CAP_PROP_POS_MSEC, float(rel_ms))
                ret, frame = cap.read()
                
                if ret:
                    if data["flip"]: frame = cv2.flip(frame, 1)
                    if data["rotate"] != 0: frame = rotate_image(frame, data["rotate"])
                    batch_frames.append(frame)
                    batch_meta.append(name)
                else:
                    w, h = writer_dims[name]
                    writers[name].write(np.zeros((h, w, 3), dtype=np.uint8))
            
            if not batch_frames:
                current_time += STEP_MS
                continue
                
            # INFERENCE
            results = model.predict(batch_frames, imgsz=640, device="cpu", conf=CONF_THRESH_OBJECT, verbose=False)
            
            for i, res in enumerate(results):
                name = batch_meta[i]
                frame = batch_frames[i]
                dt_now = cam_data[name]["start_dt"] + datetime.timedelta(milliseconds=(current_time - cam_data[name]["start_ms"])) if cam_data[name]["start_dt"] else None
                
                detections = []
                for box in res.boxes:
                    c = int(box.cls[0])
                    conf = float(box.conf[0])
                    c_name = model.model.names.get(c, str(c))
                    
                    if c_name not in WANTED: continue
                        
                    # Split Thresholds
                    thresh = CONF_THRESH_PERSON if c_name == "person" else CONF_THRESH_OBJECT
                    if conf < thresh: continue
                        
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    detections.append([x1,y1,x2,y2,conf,c_name])
                
                trackers[name].update(detections)
                tracks = trackers[name].get_tracks()
                
                for tid, bbox, trace in tracks:
                    internal_track = trackers[name].tracks.get(tid)
                    if not internal_track: continue
                    lost_count = internal_track[1]
                    
                    if lost_count > 0: continue # Ghost Fix
                    
                    label = f"ID:{tid}"
                    base_color = (200,200,200)
                    
                    best_match = None
                    max_iou = 0
                    for d in detections:
                        dx1,dy1,dx2,dy2,dconf,dcls = d
                        xA=max(bbox[0],dx1); yA=max(bbox[1],dy1)
                        xB=min(bbox[2],dx2); yB=min(bbox[3],dy2)
                        inter = max(0,xB-xA)*max(0,yB-yA)
                        if inter > max_iou:
                            max_iou = inter
                            best_match = d
                    
                    if best_match and max_iou > 0:
                         dcls = best_match[5]
                         dconf = best_match[4]
                         label = f"{dcls} {dconf:.2f}"
                         base_color = COLORS.get(dcls, COLORS["default"])
                    
                    x1,y1,x2,y2 = map(int, bbox)
                    cx,cy = (x1+x2)//2, (y1+y2)//2
                    
                    # Trail
                    if len(trace) > 1:
                        pts = np.array(trace, np.int32).reshape((-1,1,2))
                        trail_color = get_faded_color(base_color, 0.4) 
                        cv2.polylines(frame, [pts], False, trail_color, 2)
                        
                    cv2.rectangle(frame, (x1,y1),(x2,y2), base_color, 2)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)
                    
                    for zname, poly in cam_data[name]["zones"].items():
                        if point_in_poly(cx, cy, poly):
                            cv2.putText(frame, f"ALERTE {zname}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 4)
                            ts_str = dt_now.isoformat() if dt_now else f"offset_{current_time}"
                            with open(alert_log_path, "a") as af:
                                af.write(f"{ts_str},{name},{label},{zname}\n")

                for poly in cam_data[name]["zones"].values():
                    pts = np.array(poly, np.int32).reshape((-1,1,2))
                    cv2.polylines(frame, [pts], True, (0,0,255), 2)
                
                writers[name].write(frame)
                
            current_time += STEP_MS
            # Print progress less frequently (~every 200 frames / 20 seconds of video)
            if (current_time - global_start) % (200 * STEP_MS) < STEP_MS:
                 prog = (current_time - global_start)/(global_end-global_start+1e-9)*100
                 print(f"[PROG] {prog:.1f}%")

    except KeyboardInterrupt:
        pass
    finally:
        for c in caps.values(): c.release()
        for w in writers.values(): w.release()

if __name__ == "__main__":
    main()
