from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..utils.database import get_db
from ..utils.models import User
from ..utils.database import execute_query


router = APIRouter()

# @router.get('/users')
# async def get_users():
#     query = User.__table__.select()
#     users = await execute_query(query)
#     return {'users': users}


@router.post("/users/")
def create_user(user_data: dict, db: Session = Depends(get_db)):
    # Your logic to create a user
    return {"message": "User created successfully"}
