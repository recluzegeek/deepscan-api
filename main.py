from fastapi import FastAPI

# from .utils import classification
from .routers import video
from .utils.database import engine

# models.Base.metadata.create_all(bind=engine)

app = FastAPI()


app.include_router(video.router, tags=['Video'])
# app.include_router(classification.router, tags=['Classification'])
# app.include_router(user.router, tags=['User'])

