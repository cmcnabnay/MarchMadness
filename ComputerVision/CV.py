from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------
# Camera Calibration
# ---------------------------

FOCAL_LENGTH = 1200  # pixels (must be calibrated)
CAR_WIDTH = 1.8      # meters

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]
# 2=car, 3=motorcycle, 5=bus, 7=truck

# ---------------------------
# Load YOLO
# ---------------------------

model = YOLO("yolov8s.pt")

# ---------------------------
# Open Video
# ---------------------------

video_path = "dashcam.mp4"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]
    frame_center = width // 2

    results = model(frame, verbose=False)

    closest_vehicle = None
    smallest_center_offset = 999999

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_width = x2 - x1

        if box_width <= 0:
            continue

        center_x = (x1 + x2) // 2

        lane_offset = abs(center_x - frame_center)

        if lane_offset < smallest_center_offset:
            smallest_center_offset = lane_offset
            closest_vehicle = (x1, y1, x2, y2, box_width, cls)

    if closest_vehicle is not None:

        x1, y1, x2, y2, box_width, cls = closest_vehicle

        distance = (CAR_WIDTH * FOCAL_LENGTH) / box_width

        label = f"{model.names[cls]}  {distance:.1f} m"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.line(
        frame,
        (frame_center, 0),
        (frame_center, height),
        (255, 0, 0),
        2
    )

    cv2.imshow("Vehicle Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()