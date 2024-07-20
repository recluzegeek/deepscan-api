import jwt
from fastapi import HTTPException



SECRET_KEY = 'Kg2aLY8Lil56i8JbHTfO0Qf23A7Tp9AbgyyLUghRInvAqH7OKaPdPqY5EzSpMH7N'
ALGORITHM = 'HS256'

def decode_jwt(token: str):
    try:
        # Decode the JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # Return the decoded payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
