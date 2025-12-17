from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from jose import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import uuid
from ...utils.config import config
from ...utils.logger import logger
from ...models.user import UserCreate, User, UserUpdate

router = APIRouter()
security = HTTPBearer()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user storage (in production, this would use a database)
users_db: dict = {}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token_data = decode_token(credentials.credentials)
    user_id = token_data.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = users_db.get(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user

@router.post("/register", response_model=User)
async def register_user(user: UserCreate):
    """
    Register a new user
    """
    try:
        # Check if user already exists
        for existing_user_id, existing_user in users_db.items():
            if existing_user.email == user.email:
                raise HTTPException(
                    status_code=400,
                    detail="User with this email already exists"
                )

        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)

        new_user = User(
            id=user_id,
            email=user.email,
            name=user.name,
            preferred_language=user.preferred_language,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Store user with hashed password
        user_data = {
            "user": new_user,
            "hashed_password": hashed_password
        }
        users_db[user_id] = user_data

        logger.info(f"Registered new user: {user.email}")

        return new_user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")

@router.post("/login", response_model=Token)
async def login_user(user_credentials: UserLogin):
    """
    Login user and return access token
    """
    try:
        # Find user by email
        user_id = None
        user_data = None

        for existing_user_id, existing_user_data in users_db.items():
            if existing_user_data["user"].email == user_credentials.email:
                user_id = existing_user_id
                user_data = existing_user_data
                break

        if user_data is None:
            raise HTTPException(
                status_code=400,
                detail="Incorrect email or password"
            )

        # Verify password
        if not verify_password(user_credentials.password, user_data["hashed_password"]):
            raise HTTPException(
                status_code=400,
                detail="Incorrect email or password"
            )

        # Create access token
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={"sub": user_id, "email": user_credentials.email},
            expires_delta=access_token_expires
        )

        logger.info(f"User logged in: {user_credentials.email}")

        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging in user: {str(e)}")

@router.get("/me", response_model=User)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user's profile
    """
    try:
        logger.info(f"Retrieved profile for user: {current_user.email}")
        return current_user
    except Exception as e:
        logger.error(f"Error retrieving user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user profile: {str(e)}")

@router.put("/me", response_model=User)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update current user's profile
    """
    try:
        user_id = None
        for existing_user_id, existing_user_data in users_db.items():
            if existing_user_data["user"].id == current_user.id:
                user_id = existing_user_id
                break

        if user_id is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Update user fields
        if user_update.name is not None:
            users_db[user_id]["user"].name = user_update.name
        if user_update.preferred_language is not None:
            users_db[user_id]["user"].preferred_language = user_update.preferred_language

        users_db[user_id]["user"].updated_at = datetime.now()

        logger.info(f"Updated profile for user: {current_user.email}")

        return users_db[user_id]["user"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating user profile: {str(e)}")