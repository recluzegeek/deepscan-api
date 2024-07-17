from fastapi import APIRouter

router = APIRouter()

@router.get('/video')
def upload_video():
    return {"video": "video uploaded"}