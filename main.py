from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

model = YOLO("runs/train/blueprint_yolo_local/weights/best.pt")
app = FastAPI()

LABELS = ['door', 'window']

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    results = model(img_array)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        detections.append({
            "label": LABELS[cls_id],
            "confidence": round(confidence, 2),
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    return {"detections": detections}
