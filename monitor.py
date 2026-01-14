import cv2
import glob
import math
import numpy as np
import time

def make_grid(images, cols=2):
    if not images:
        return None
    n = len(images)
    rows = math.ceil(n / cols)
    
    # Resize all to common height (e.g., 360px) to keep ratio
    target_h = 360
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(img, (int(w * scale), target_h)))
        
    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))
        
    # Concat
    grid_rows = []
    for r in range(rows):
        row_imgs = resized[r*cols : (r+1)*cols]
        # Ensure widths match in row? With fixed height, widths might vary if aspect ratios differ.
        # Simple approach: fixed size for all? Or center in fixed box?
        # Let's just use hconcat if widths are same. If not, we might have issues.
        # Assuming most videos are 16:9 or similar.
        # To be safe, let's resize to fixed WxH.
        row_imgs_fixed = [cv2.resize(img, (640, 360)) for img in row_imgs]
        grid_rows.append(cv2.hconcat(row_imgs_fixed))
        
    return cv2.vconcat(grid_rows)

def main():
    while True:
        files = sorted(glob.glob("output/*_annotated.mp4"))
        if not files:
            print("Waiting for files...")
            time.sleep(1)
            continue
            
        print(f"Found {len(files)} files. Starting playback...")
        caps = [cv2.VideoCapture(f) for f in files]
        names = [f.split("/")[-1] for f in files]
        
        while True:
            frames = []
            active = False
            
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    # Rewind or show black?
                    # Let's rewind to loop independently
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                if ret:
                    active = True
                    # Put name
                    cv2.putText(frame, names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((360, 640, 3), dtype=np.uint8))
            
            if not active:
                print("All streams ended (or failed).")
                break
                
            grid = make_grid(frames, cols=3)
            if grid is not None:
                # Resize to fit screen if huge
                h, w = grid.shape[:2]
                if h > 800:
                    scale = 800 / h
                    grid = cv2.resize(grid, (int(w*scale), 800))
                    
                cv2.imshow("Monitoring MVP", grid)
            
            if cv2.waitKey(100) & 0xFF == ord('q'): # 100ms = 10fps playback for viewing
                for c in caps: c.release()
                return

if __name__ == "__main__":
    main()
