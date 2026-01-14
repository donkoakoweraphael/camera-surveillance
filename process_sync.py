# process_sync.py
import os, json, time, math, subprocess, datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from sort import SimpleTracker

# CONFIG
CAMS_FILE = "cams.json"
OUTPUT_DIR = "output"
MODEL_NAME = "yolov8n.pt"   # ultralytics téléchargera automatiquement
TARGET_FPS = 2              # fps de traitement par caméra (2 = une frame tous les 500ms)
CONF_THRESH = 0.35

# classes we want: person, backpack/handbag, bottle
WANTED = ["person", "backpack", "handbag", "bottle"]

def ensure(d):
    os.makedirs(d, exist_ok=True)

def read_cams():
    with open(CAMS_FILE) as f:
        return json.load(f)

def ffprobe_start_time(path):
    # try to get creation_time from metadata
    cmd = [
        "ffprobe","-v","quiet","-print_format","json",
        "-show_format","-show_streams", str(path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        j = json.loads(res.stdout)
        # try format.tags.creation_time
        fmt = j.get("format", {})
        tags = fmt.get("tags", {}) or {}
        ct = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
        if ct:
            # ffprobe returns like "2025-01-14T09:00:00.000000Z" or similar
            try:
                dt = datetime.datetime.fromisoformat(ct.replace("Z",""))
            except:
                dt = None
            return dt
    except Exception:
        return None
    return None

def get_video_duration_ms(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration_s = frame_count / fps if fps>0 else 0
    return int(duration_s * 1000)

def time_to_ms(dt):
    # dt: datetime -> ms since epoch
    return int(dt.timestamp()*1000)

def align_and_process(cams):
    ensure(OUTPUT_DIR)
    # determine start_time for each cam (ms since epoch)
    cam_info = {}
    for name, cfg in cams.items():
        path = Path(cfg["file"])
        if not path.exists():
            print(f"[ERR] {name} file not found: {path}")
            continue
        # try ffprobe
        dt = ffprobe_start_time(path)
        if not dt:
            s = cfg.get("start_time")
            if s:
                try:
                    dt = datetime.datetime.fromisoformat(s)
                except:
                    dt = None
        if not dt:
            print(f"[WARN] {name}: no start_time found. Will assume epoch 0 for this cam. (add start_time in cams.json for proper sync)")
            start_ms = 0
        else:
            start_ms = time_to_ms(dt)
        dur_ms = get_video_duration_ms(path)
        cam_info[name] = {
            "path": str(path),
            "flip": cfg.get("flip", False),
            "zones": cfg.get("zones", {}),
            "start_ms": start_ms,
            "duration_ms": dur_ms
        }
        print(f"[INFO] {name}: start_ms={start_ms}, duration_ms={dur_ms}")
    # compute global window
    starts = [v["start_ms"] for v in cam_info.values() if v["duration_ms"]]
    ends = [v["start_ms"] + v["duration_ms"] for v in cam_info.values() if v["duration_ms"]]
    if not starts:
        print("Aucune video valide trouvee.")
        return
    global_start = min(starts)
    global_end = max(ends)
    print(f"[INFO] global window ms: {global_start} .. {global_end}")

    # init model and trackers
    model = YOLO(MODEL_NAME)
    trackers = {name: SimpleTracker() for name in cam_info.keys()}
    writers = {}
    caps = {}

    # prepare VideoWriters
    for name, info in cam_info.items():
        cap = cv2.VideoCapture(info["path"])
        caps[name] = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outpath = os.path.join(OUTPUT_DIR, f"{name}_annotated.mp4")
        writers[name] = cv2.VideoWriter(outpath, fourcc, TARGET_FPS, (w,h))
        print(f"[OUT] {name} -> {outpath}")

    step_ms = int(1000 / TARGET_FPS)
    current = global_start
    alert_log = os.path.join(OUTPUT_DIR, "alerts.log")

    while current <= global_end:
        for name, info in cam_info.items():
            cap = caps[name]
            start_ms = info["start_ms"]
            rel_ms = current - start_ms  # relative ms within video
            if rel_ms < 0 or rel_ms > info["duration_ms"]:
                # no frame at this global time for this camera
                continue
            # seek to rel_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, float(rel_ms))
            ret, frame = cap.read()
            if not ret:
                continue
            if info["flip"]:
                frame = cv2.flip(frame, 1)
            # Run detection (YOLOv8) on CPU, resized inside model
            res = model.predict(source=frame, imgsz=640, device="cpu", conf=CONF_THRESH, verbose=False)
            # res is list with one element
            detections = []
            r = res[0]
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # map class id to name
                name_cls = model.model.names.get(cls_id, str(cls_id))
                if name_cls not in WANTED:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                detections.append([x1,y1,x2,y2,conf,name_cls])
            # update tracker
            trackers[name].update(detections)
            tracks = trackers[name].get_tracks()
            # annotate frame
            for tid, bbox, trace in tracks:
                x1,y1,x2,y2 = [int(v) for v in bbox]
                cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"ID{tid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                # draw trace
                for i in range(1, len(trace)):
                    cv2.line(frame, tuple(trace[i-1]), tuple(trace[i]), (255,0,0), 2)
                # zone check
                for zn, poly in info["zones"].items():
                    if point_in_poly(cx, cy, poly):
                        cv2.putText(frame, f"ALERTE {zn}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),3)
                        with open(alert_log, "a") as f:
                            f.write(f"{datetime.datetime.utcnow().isoformat()},{name},{tid},{zn},{cx},{cy}\n")
            # draw zones
            for zn, poly in info["zones"].items():
                pts = np.array(poly, np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,0,255), 2)
            # write annotated frame to writer
            writers[name].write(frame)
        current += step_ms

    # release
    for c in caps.values(): c.release()
    for w in writers.values(): w.release()
    print("Traitement termine. Sorties dans ./output/")
    return

# point_in_poly copied here
def point_in_poly(x,y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i+1)%n]
        intersect = ((yi>y) != (yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi)
        if intersect:
            inside = not inside
    return inside

if __name__ == "__main__":
    cams = json.load(open(CAMS_FILE))
    align_and_process(cams)
