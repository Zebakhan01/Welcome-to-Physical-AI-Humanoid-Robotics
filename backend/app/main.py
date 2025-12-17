from fastapi import FastAPI
from app.api import chat

app = FastAPI(
    title="RAG Chatbot API",
    description="AI-Hackathon RAG Chatbot Backend",
    version="0.1.0"
)

# Include API routes
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "AI-Hackathon RAG Chatbot Backend", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)