import cv2
from ultralytics import YOLO

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "yolo11n.pt")
model = YOLO(model_path, verbose=False)

def process_frame(img):
    results = model(img, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            color = (0, int(conf * 255), int((1 - conf) * 255))
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # cv2.imshow("Processed Frame", img)
    # cv2.waitKey(1)

    return img
