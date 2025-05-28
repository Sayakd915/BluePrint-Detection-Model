from ultralytics import YOLO
import cv2

model = YOLO("runs/train/blueprint_yolo_local/weights/best.pt")

test_img = "images/1_png.rf.16a0f4557e684635c98d47dbd355e0bf.jpg" 

results = model(test_img, save=True, imgsz=640, conf=0.25)

print("Inference done. Check 'runs/detect/predict/'")
