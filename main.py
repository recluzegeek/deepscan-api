from fastapi import FastAPI

from .routers import video_router
from .utils.database import engine, Base

app = FastAPI()

# Create the database tables
Base.metadata.create_all(bind=engine)

app.include_router(video_router.router, tags=['Video'])
