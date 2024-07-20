from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .routers import classification, video, user
from .utils.database import engine
# from .utils.middleware import UserIDMiddleware


# models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add the custom middleware
# app.add_middleware(UserIDMiddleware)

@app.middleware('http')
async def transform_user_id(request: Request, call_next):
    if request.method == 'POST':
        
        # Read the body as bytes
        body = await request.body()
        
        if not body:
            return JSONResponse(
                content={"detail": "Request body is required"},
                status_code=400
            )


        body = await request.json()
        user_id = body.get('user_id')
                
        if user_id is None:
                    # Return a JSON response for missing user_id
            return JSONResponse(
                content={"detail": "user_id is required in the request body"},
                        status_code=400
            )

    # Call the next middleware or endpoint
    response = await call_next(request)
    return response



app.include_router(user.router, tags=['User'])
app.include_router(video.router, tags=['Video'])
app.include_router(classification.router, tags=['Classification'])

@app.post('/{name}')
def hello(name):
    return {"hello": "world by {name}"}