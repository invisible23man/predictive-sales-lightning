from fastapi import FastAPI

from src.app.api import health, predict

app = FastAPI()
app.include_router(health.router, prefix="/system")
app.include_router(predict.router, prefix="/api")
