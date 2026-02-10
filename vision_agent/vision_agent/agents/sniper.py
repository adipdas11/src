from ultralytics import YOLO
import numpy as np
import cv2

class SniperAgent:
    def __init__(self, model_path):
        print(f"🎯 Loading Local Sniper (Pose): {model_path}")
        self.model = YOLO(model_path)
    
    def target(self, frame):
        """
        Returns structured dict with Screw_Heads, Tool_Tips, and Holes.
        """
        results = self.model(frame, verbose=False)
        
        # New Structured Output
        data = {
            "screw_heads": [],
            "tool_tips": [],
            "holes": []
        }
        
        for r in results:
            if r.boxes is None: continue
            
            # Get data arrays
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            # Check if keypoints exist
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy() # Shape: (N, num_kpts, 2)
            else:
                kpts = None

            for i, cls_id in enumerate(classes):
                label = r.names[int(cls_id)]
                box = boxes[i].astype(int).tolist()
                conf = float(confs[i])
                
                # Base object info
                obj = {
                    "label": label,
                    "conf": round(conf, 3),
                    "box": box
                }

                # --- CLASSIFY THE DOTS ---
                if label == "Screw_Head":
                    # For Screws, we want the Center keypoint
                    if kpts is not None and len(kpts[i]) > 0:
                        cx, cy = kpts[i][0]
                        if cx != 0 and cy != 0: 
                            obj["center"] = [int(cx), int(cy)]
                    data["screw_heads"].append(obj)

                elif label == "Tool_Tip":
                    # For Tools, we want the Contact Point
                    if kpts is not None and len(kpts[i]) > 0:
                        cx, cy = kpts[i][0]
                        if cx != 0 and cy != 0:
                            obj["contact_point"] = [int(cx), int(cy)]
                    data["tool_tips"].append(obj)

                elif label == "Hole":
                    # Holes might not have keypoints, just boxes
                    data["holes"].append(obj)

        return data