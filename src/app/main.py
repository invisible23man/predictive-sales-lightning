from fastapi import FastAPI
from src.app.api import health

app = FastAPI()
app.include_router(health.router)
