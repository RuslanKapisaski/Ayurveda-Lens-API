from fastapi import FastAPI
from app.routes.food import router

app = FastAPI(
    title="Auyrveda Lens API",
    description="AI powered food recognision with Auyrveda recommendation",
    version="1.0",
)

app.include_router(router,prefix="/api")

@app.get('/')
def root():
    return {"status": "Ayurveda Lens is running.."}
