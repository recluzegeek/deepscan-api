from fastapi import FastAPI
from .routers import classification, video
from .utils.database import engine
from .utils import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(video.router, tags=['Video'])
app.include_router(classification.router, tags=['Classification'])

@app.get('/')
def welcome():
    return {"details": "welcome to deepscan api"}