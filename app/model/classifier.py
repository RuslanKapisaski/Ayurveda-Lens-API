import os
import requests
from dotenv import load_dotenv
from firebase_admin import firestore

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/nateraw/food"
LOW_CONFIDENCE_THRESHOLD = 80

firestore_db = firestore.client()


def get_food_profile(food_name: str):
    food_doc = (
        firestore_db
        .collection("foods")
        .document(food_name)
        .get()
    )

    if not food_doc.exists:
        return {
            "name": food_name,
            "dosha": {
                "vata": "neutral",
                "pitta": "neutral",
                "kapha": "neutral",
            },
            "explanation": (
                f"No detailed Ayurveda profile was found for {food_name}. "
                "The food is treated as neutral."
            ),
            "alternatives": [],
            "allergens": [],
            "properties": [],
            "isActive": False,

        }

    return food_doc.to_dict()


def classify_image(image_bytes: bytes):
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(
        API_URL,
        headers=headers,
        data=image_bytes,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"HF API error: {response.status_code} {response.text}"
        )

    results = response.json()

    if not results:
        raise RuntimeError("Empty response from Hugging Face")

    top_result = results[0]

    food_name = top_result["label"].lower().replace("_", " ")
    confidence = round(top_result["score"] * 100, 2)

    food_data = get_food_profile(food_name)

    if not food_data.get("isActive", False):
        return {
            "food": food_name,
            "confidence": confidence,
            "requires_feedback": True,
            "status": "food_not_found",
        }

    return {
        "food": food_name,
        "confidence": confidence,
        "dosha": food_data.get("dosha"),
        "explanation": food_data.get("explanation"),
        "alternatives": food_data.get("alternatives", []),
        "allergens": food_data.get("allergens", []),
        "properties": food_data.get("properties", []),
        "status": (
            "low_confidence"
            if confidence < LOW_CONFIDENCE_THRESHOLD
            else "success"
        ),

    }
