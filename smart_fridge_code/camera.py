import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Smart Fridge Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()