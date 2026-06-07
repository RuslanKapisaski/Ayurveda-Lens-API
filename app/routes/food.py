from fastapi import APIRouter, File, UploadFile, Depends

from app.auth.firebase_auth import get_current_user
from app.model.classifier import classify_image

router = APIRouter()

@router.post("/scan")
async def scan_food(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    image_bytes = await file.read()

    result = classify_image(image_bytes)

    return {
       **result,
       "userId": current_user["uid"],
    }