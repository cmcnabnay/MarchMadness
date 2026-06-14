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

TIME_GAP = 2.0  # seconds
MPS_TO_MPH = 2.23694

# ---------------------------
# Load YOLO
# ---------------------------

model = YOLO("yolov8s.pt")

# ---------------------------
# Open Video
# ---------------------------

video_path = "C:\\Users\\cmcna\\Git Repositories\\ComputerVision\\8357-208052048.mp4"

cap = cv2.VideoCapture(video_path)
print("Video opened:", cap.isOpened())

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

        recommended_speed_mps = distance / TIME_GAP
        recommended_speed_mph = recommended_speed_mps * MPS_TO_MPH

        label = (
            f"{distance:.1f} m | "
            f"Safe Speed: {recommended_speed_mph:.1f} mph"
        )

        cv2.rectangle(
            frame,
            (20, 20),
            (450, 120),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.1f} m",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Recommended Speed: {recommended_speed_mph:.1f} mph",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    cv2.imshow("Vehicle Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()