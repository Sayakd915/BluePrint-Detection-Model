from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="blueprint_yolo_local",
    project="runs/train",
    augment=True,
    hsv_h=0.015,       
    hsv_s=0.7,         
    hsv_v=0.4,         
    degrees=0.0,       
    translate=0.1,     
    scale=0.5,         
    shear=0.2,         
    perspective=0.001, 
    flipud=0.1,        
    fliplr=0.5,        
    mosaic=1.0,        
    mixup=0.1,        
    copy_paste=0.2,
    patience=15,
    verbose=True
)

run_dir = model.trainer.save_dir  # Directory like runs/train/blueprint_yolo_local
metrics_file = os.path.join(run_dir, "results.csv")