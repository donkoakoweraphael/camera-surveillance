import cv2
import json
import numpy as np

CAMS_FILE = "cams.json"

def load_cams():
    with open(CAMS_FILE, 'r') as f:
        return json.load(f)

def save_cams(data):
    with open(CAMS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def click_event(event, x, y, flags, params):
    # params = (current_points_list, image_to_draw_on)
    pts = params[0]
    img = params[1]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.line(img, tuple(pts[-2]), tuple(pts[-1]), (0, 255, 0), 2)
        cv2.imshow("Zone Selector", img)

def select_zone(image, window_name="Zone Selector"):
    points = []
    # Work on copy
    img_copy = image.copy()
    
    cv2.imshow(window_name, img_copy)
    cv2.setMouseCallback(window_name, click_event, param=(points, img_copy))
    
    print("  -> Click points to define polygon.")
    print("  -> Press 'c' to clear.")
    print("  -> Press 's' to save/finish.")
    print("  -> Press 'q' to skip/quit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Close loop
            if len(points) > 2:
                # draw closing line
                cv2.line(img_copy, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
                cv2.waitKey(500)
            break
        elif key == ord('c'):
            points = []
            img_copy = image.copy()
            cv2.imshow(window_name, img_copy)
            cv2.setMouseCallback(window_name, click_event, param=(points, img_copy))
        elif key == ord('q'):
            return None
            
    return points

def main():
    data = load_cams()
    
    print("====================================")
    print("       ZONE SELECTOR TOOL           ")
    print("====================================")
    
    for name, cfg in data.items():
        print(f"\nProcessing Camera: {name}")
        path = cfg["file"]
        
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"[ERR] Could not read {path}")
            continue
            
        # Apply transforms just like pipeline
        if cfg.get("flip", False):
            frame = cv2.flip(frame, 1)
        
        rot = cfg.get("rotate", 0)
        if rot == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Determine current zones
        zones = cfg.get("zones", {})
        
        while True:
            action = input(f"Current zones: {list(zones.keys())}. Add new zone? (y/n): ").strip().lower()
            if action != 'y':
                break
                
            zname = input("Enter zone name (e.g. zone_alerte, entry, exit): ").strip()
            if not zname: continue
            
            print(f"Defining {zname}...")
            poly = select_zone(frame)
            
            if poly:
                zones[zname] = poly
                print(f"Saved {zname} with {len(poly)} points.")
            else:
                print("Skipped.")
                
        # Update config
        cfg["zones"] = zones
        save_cams(data)
        print("Progress saved to cams.json")
        
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
