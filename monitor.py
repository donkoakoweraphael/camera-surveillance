import cv2
import glob
import math
import numpy as np
import time

def add_letterbox(img, target_size):
    # Resize to fit within target_size (width, height) maintaining aspect ratio
    # Then pad to fill the target_size (centering the image)
    
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    
    # Resize
    if scale != 1.0:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img
    
    # Calculate padding for centering
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    
    # return with border
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

def make_grid(images, cols=4, cell_size=(1280, 1280), margin=20):
    if not images: return None
    
    # Process each image to fit into the uniform cell_size
    processed = [add_letterbox(img, cell_size) for img in images]
    
    # Fill missing cells with black
    rows = math.ceil(len(processed) / cols)
    required_cells = rows * cols
    while len(processed) < required_cells:
        processed.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))
        
    # Create the Canvas
    total_w = cols * cell_size[0] + (cols + 1) * margin
    total_h = rows * cell_size[1] + (rows + 1) * margin
    
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(processed):
        r = idx // cols
        c = idx % cols
        
        y = margin + r * (cell_size[1] + margin)
        x = margin + c * (cell_size[0] + margin)
        
        canvas[y:y+cell_size[1], x:x+cell_size[0]] = img
        
    return canvas

def main():
    print("Starting Monitor High-Res...")
    cv2.namedWindow("Surveillance Monitor v6 High-Res", cv2.WINDOW_NORMAL)
    # Hint: Start window maximized or reasonable size might be needed, user can resize.
    
    while True:
        files = sorted(glob.glob("output/*_annotated.mp4"))
        if not files:
            print("Waiting for files...")
            time.sleep(1)
            continue
            
        print(f"Loading {len(files)} streams...")
        caps = [cv2.VideoCapture(f) for f in files]
        names = [f.split("/")[-1].replace("_annotated.mp4","") for f in files]
        
        max_frames = 0
        for c in caps:
            fc = c.get(cv2.CAP_PROP_FRAME_COUNT)
            if fc > max_frames: max_frames = int(fc)
            
        current_frame = 0
        paused = False
        
        # Determine Cell Size
        # User requested large resolution (1280x720 or 720x1280).
        # A cell size of 1280x1280 accommodates both portrait and landscape fully.
        CELL_W = 1280
        CELL_H = 1280
        
        while True:
            # Sync
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            frames = []
            active = False
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    active = True
                    # Re-add name if needed (pipeline adds labels, but monitor overlay is nice)
                    # Pipeline already burns text? "names[i]" passed to pipeline?
                    # Pipeline draws "ALERTE" etc.
                    # We can add clean Camera Name overlay here too.
                    cv2.putText(frame, names[i], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((720, 1280, 3), dtype=np.uint8)) # Placeholder

            if not active:
                current_frame = 0
                continue

            # Build Grid
            grid = make_grid(frames, cols=4, cell_size=(CELL_W, CELL_H), margin=20)
            
            # Progress Bar
            ph, pw = grid.shape[:2]
            bar_w = int((current_frame / max_frames) * pw) if max_frames > 0 else 0
            cv2.rectangle(grid, (0, ph-30), (bar_w, ph), (0, 255, 0), -1)
            
            cv2.imshow("Surveillance Monitor v6 High-Res", grid)
            
            wait_time = 1000 if not paused else 100
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                for c in caps: c.release()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == 81 or key == ord('a'): 
                current_frame = max(0, current_frame - 1)
            elif key == 83 or key == ord('d'): 
                current_frame = min(max_frames-1, current_frame + 1)
            else:
                 if not paused:
                     current_frame += 1
                     if current_frame >= max_frames: current_frame = 0

if __name__ == "__main__":
    main()
