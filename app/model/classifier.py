import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/nateraw/food"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSHA_MAP_PATH = os.path.join(BASE_DIR, "../data/dosha_map.json")

with open(DOSHA_MAP_PATH, "r") as f:
    dosha_map = json.load(f)


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
        raise RuntimeError(f"HF API error: {response.status_code} {response.text}")

    results = response.json()
    top_result = results[0]

    food_name = top_result["label"].lower().replace("_", " ")
    confidence = round(top_result["score"] * 100, 2)

    dosha = dosha_map.get(food_name, {
        "vata": "neutral",
        "pitta": "neutral",
        "kapha": "neutral",
    })

    return {
        "food": food_name,
        "confidence": confidence,
        "dosha": dosha,
    }