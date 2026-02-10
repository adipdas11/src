from ultralytics import YOLO
import numpy as np
import cv2

class ScoutAgent:
    def __init__(self, model_path):
        print(f"🚀 Loading Global Scout (Seg): {model_path}")
        self.model = YOLO(model_path)
    
    def scan(self, frame):
        """Returns list of detected objects with bounding boxes, MASKS, and debug image."""
        results = self.model(frame, verbose=False)
        detections = []
        
        # Draw standard debug frame (Optional, can be removed if relying on agent_node vis)
        debug_frame = results[0].plot()

        for r in results:
            # Check if we have boxes
            if r.boxes is None:
                continue
                
            # Iterate through detections
            for i, box in enumerate(r.boxes):
                # 1. Basic Box Info
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                obj = {
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "segments": [] # Default to empty
                }

                # 2. EXTRACT MASKS (The Missing Piece)
                if r.masks is not None:
                    # r.masks.xy is a list of arrays. Each array = polygon points for one object.
                    # They correspond by index to r.boxes.
                    if len(r.masks.xy) > i:
                        # Get the points (N, 2)
                        poly_points = r.masks.xy[i]
                        
                        # Filter out bad/empty masks (needs at least 3 points for a polygon)
                        if len(poly_points) >= 3:
                            # Convert numpy array to simple list of lists for JSON
                            obj["segments"] = poly_points.astype(int).tolist()

                detections.append(obj)
                
        return detections, debug_frame