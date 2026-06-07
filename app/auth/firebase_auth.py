import os
import json
import firebase_admin
from fastapi import Depends, HTTPException
from firebase_admin import credentials, auth
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def init_firebase():
    if firebase_admin._apps:
        return

    firebase_credentials = os.environ.get("FIREBASE_CREDENTIALS")

    if not firebase_credentials:
        raise RuntimeError("Missing FIREBASE_CREDENTIALS environment variable")

    cred_dict = json.loads(firebase_credentials)
    cred = credentials.Certificate(cred_dict)

    firebase_admin.initialize_app(cred)
    print("Firebase Admin initialized successfully")


init_firebase()

async def get_current_user(credentials: HTTPAuthorizationCredentials=Depends(security)):
    token = credentials.credentials

    try:
        decoded_token = auth.verify_id_token(token)

        return {
            "uid":decoded_token["uid"],
            "email":decoded_token["email"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Firebase token",
        )