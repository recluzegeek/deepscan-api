from fastapi import FastAPI

from .routers import video
from .utils.database import engine, Base

app = FastAPI()

# Create the database tables
Base.metadata.create_all(bind=engine)

app.include_router(video.router, tags=['Video'])
