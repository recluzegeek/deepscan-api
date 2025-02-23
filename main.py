from fastapi import FastAPI

from .routers import video_router

app = FastAPI()

app.include_router(video_router.router, tags=['Video'])
