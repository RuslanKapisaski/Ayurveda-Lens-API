from fastapi import APIRouter, File, UploadFile
from app.model.classifier import classify_image

router = APIRouter()

@router.post("/scan")
async def scan_food(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result = classify_image(image_bytes)

    return {
        "food": result["food"],
        "confidence": result["confidence"],
        "dosha_recommendation": result["dosha"],
    }