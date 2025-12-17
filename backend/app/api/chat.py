from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: list = []

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Placeholder for chat functionality
    return {"response": "Chat endpoint is working", "message": request.message}