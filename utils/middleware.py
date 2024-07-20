# https://semaphoreci.com/blog/custom-middleware-fastapi

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from .database import get_db
from .models import User
from fastapi.responses import JSONResponse

class UserIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Check if the request method is one that typically has a body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                user_id = body.get("user_id")
                
                if user_id is None:
                    # Return a JSON response for missing user_id
                    return JSONResponse(
                        content={"detail": "user_id is required in the request body"},
                        status_code=400
                    )

                # Get a database session
                db: Session = next(get_db())
                # Verify if user_id exists in the database
                user = db.query(User).filter(User.id == user_id).first()

                if user is None:
                    # Return a JSON response for non-existing user_id
                    return JSONResponse(
                        content={"detail": "User does not exist"},
                        status_code=404
                    )

            except Exception as e:
                # Handle any unexpected errors gracefully
                return JSONResponse(
                    content={"detail": "An error occurred"},
                    status_code=500
                )

        # Proceed with the request
        response = await call_next(request)
        return response
