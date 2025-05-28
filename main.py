from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI()
model = None 
LABELS = ['door', 'window']

def load_model():
    global model
    if model is None:
        model = YOLO("runs/train/blueprint_yolo_local/weights/best.pt")
    return model

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    current_model = load_model()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((640, 640))
    img_array = np.array(img)

    results = current_model(img_array)[0]
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)