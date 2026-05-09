from transformers import pipeline
import json
import os

# Load dosha map
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSHA_MAP_PATH = os.path.join(BASE_DIR, '../data/dosha_map.json')

with open(DOSHA_MAP_PATH, 'r') as f:
    dosha_map = json.load(f)

# Food101 pretrained model
classifier = pipeline(
    task="image-classification",
    model="nateraw/food"
)

def classify_image(tensor):
    results = classifier(tensor,top_k=1)
    top_result = results[0]

    food_name = top_result['label'].lower().replace("_"," ")
    confidence = round(top_result['score']*100, 2)

    dosha = dosha_map.get(food_name,{
        "vata": "neutral",
        "pitta": "neutral",
        "kapha": "neutral",
    })

    return {
        "food": food_name,
        "confidence": confidence,
        "dosha": dosha
    }