import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="MeatAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model YOLO (pastikan file best.pt ada di folder yang sama)
model = YOLO("best.pt")

@app.get("/")
def root():
    return {"status": "online", "message": "MeatAI API Ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference di CPU
    results = model.predict(source=image, conf=0.35, device="cpu")

    predictions = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            predictions.append({
                "label": class_name,
                "confidence": round(confidence, 4)
            })

    if not predictions:
        return {"success": True, "detected": False, "message": "Daging tidak terdeteksi"}

    top_prediction = max(predictions, key=lambda x: x["confidence"])

    return {
        "success": True,
        "detected": True,
        "main_result": top_prediction["label"],
        "details": predictions
    }
