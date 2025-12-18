from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from uuid import UUID
import uuid
from ...database import get_db
from ...models.sqlalchemy_models import User
from ...services.database_service import get_user_by_email, create_user, update_user

router = APIRouter()

class UserPreferences(BaseModel):
    preferred_language: str = "en"
    learning_level: str = "intermediate"  # beginner, intermediate, advanced
    notification_enabled: bool = True
    theme_preference: str = "light"  # light, dark, auto

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    preferred_language: Optional[str] = None
    learning_level: Optional[str] = None
    notification_enabled: Optional[bool] = None
    theme_preference: Optional[str] = None

class UserRegistrationRequest(BaseModel):
    email: str
    name: Optional[str] = None
    preferred_language: str = "en"

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    preferred_language: str
    learning_level: str
    notification_enabled: bool
    theme_preference: str
    created_at: str

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserRegistrationRequest,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    """
    try:
        # Check if user already exists
        existing_user = get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        # Create new user
        db_user = create_user(
            db=db,
            user_data=user_data
        )

        return UserResponse(
            id=str(db_user.id),
            email=db_user.email,
            name=db_user.name,
            preferred_language=db_user.preferred_language,
            learning_level="intermediate",  # Default value
            notification_enabled=True,      # Default value
            theme_preference="light",       # Default value
            created_at=db_user.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User registration failed: {str(e)}")

@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    user_id: str,  # In a real app, this would come from auth middleware
    db: Session = Depends(get_db)
):
    """
    Get user profile information
    """
    try:
        db_user = db.query(User).filter(User.id == UUID(user_id)).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=str(db_user.id),
            email=db_user.email,
            name=db_user.name,
            preferred_language=db_user.preferred_language,
            learning_level="intermediate",  # Would come from user preferences table in a full implementation
            notification_enabled=True,      # Would come from user preferences
            theme_preference="light",       # Would come from user preferences
            created_at=db_user.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user profile: {str(e)}")

@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_id: str,  # In a real app, this would come from auth middleware
    profile_update: UserProfileUpdate,
    db: Session = Depends(get_db)
):
    """
    Update user profile information
    """
    try:
        # Update user information
        updated_user = update_user(
            db=db,
            user_id=UUID(user_id),
            user_update=profile_update
        )

        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=str(updated_user.id),
            email=updated_user.email,
            name=updated_user.name,
            preferred_language=updated_user.preferred_language,
            learning_level="intermediate",  # Would be updated from profile_update in a full implementation
            notification_enabled=True,      # Would be updated from profile_update
            theme_preference="light",       # Would be updated from profile_update
            created_at=updated_user.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user profile: {str(e)}")

@router.get("/preferences")
async def get_user_preferences(
    user_id: str,  # In a real app, this would come from auth middleware
    db: Session = Depends(get_db)
):
    """
    Get user preferences
    """
    try:
        db_user = db.query(User).filter(User.id == UUID(user_id)).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserPreferences(
            preferred_language=db_user.preferred_language,
            learning_level="intermediate",      # Would come from preferences table
            notification_enabled=True,          # Would come from preferences table
            theme_preference="light"            # Would come from preferences table
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user preferences: {str(e)}")

@router.put("/preferences")
async def update_user_preferences(
    user_id: str,  # In a real app, this would come from auth middleware
    preferences: UserPreferences,
    db: Session = Depends(get_db)
):
    """
    Update user preferences
    """
    try:
        # Update the user's preferred language in the main user table
        db_user = db.query(User).filter(User.id == UUID(user_id)).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update the preferred language in the user table
        db_user.preferred_language = preferences.preferred_language
        db.commit()
        db.refresh(db_user)

        # In a full implementation, we would store other preferences in a separate preferences table
        return preferences
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user preferences: {str(e)}")