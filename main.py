from fastapi import FastAPI

from .routers import video
from .utils.database import engine

app = FastAPI()


app.include_router(video.router, tags=['Video'])
