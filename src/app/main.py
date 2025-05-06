from fastapi import FastAPI

from src.app.api import health, predict, train

app = FastAPI()

app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(train.router, prefix="/api")
