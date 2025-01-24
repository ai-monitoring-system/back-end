from ultralytics import YOLO
from collections import deque
import cv2
import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "yolo11n.pt")
model = YOLO(model_path, verbose=False)

# Store areas for 5 seconds
area_history = deque(maxlen=150)

def process_frame(img):
    results = model(img, verbose=False)
    current_area = 0

    for result in results:
        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            color = (0, int(conf * 255), int((1 - conf) * 255))
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Find largest person in frame
        person_boxes = [box for box in result.boxes if model.names[int(box.cls)] == "person"]
        if person_boxes:
            # Get largest person
            largest_person = max(person_boxes, key=lambda box: (box.xyxy[0][2]-box.xyxy[0][0])*(box.xyxy[0][3]-box.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_person.xyxy[0])
            current_area = (x2 - x1) * (y2 - y1)
            area_history.append(current_area)

    # Check for approaching person
    if len(area_history) >= 30:
        is_growing, growth_rate = check_area_growth(area_history)
        if is_growing:
            print(f"Person is approaching! Growth rate: {growth_rate:.2f} px²/frame")

    return img

def check_area_growth(history):
    # Use last N samples for analysis
    y = np.array(history)
    x = np.arange(len(y))
    
    # Perform linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate R² value
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return (slope > 50 and r_squared > 0.5), slope