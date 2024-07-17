from pydantic import BaseModel


class Videos(BaseModel):
    video_id: str
    user_id: str
    