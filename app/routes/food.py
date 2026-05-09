from fastapi import APIRouter, File, UploadFile
from app.model.classifier import classify_image
from PIL import Image
import io

router = APIRouter()

@router.post("/scan")
async def scan_food(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = classify_image(image)

    return {
        'food':result["food"],
        'confidence':result["confidence"],
        'dosha_recommendation':result["dosha"],
    }

