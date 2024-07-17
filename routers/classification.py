from fastapi import APIRouter

router = APIRouter()

@router.get('/class')
def classification():
    return {"classification": "classifying video"}