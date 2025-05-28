from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os

model = YOLO("yolov8s.pt")
model.train(
    data="dataset.yaml",
    epochs=75,
    imgsz=640,
    batch=4,
    name="blueprint_yolo_local",
    project="runs/train",
    augment=True,
    degrees=15,
    translate=0.1,
    scale=0.5,
    shear=10,
    mosaic=True,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=20,
    verbose=True
)

run_dir = model.trainer.save_dir  # Directory like runs/train/blueprint_yolo_local
metrics_file = os.path.join(run_dir, "results.csv")