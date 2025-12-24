"""
Startup script for Phase 2 RAG Backend
"""
import os
import sys
from dotenv import load_dotenv
import uvicorn
from .config.rag_settings import settings


def main():
    """Start the backend server"""
    # Load environment variables
    load_dotenv()

    print("🚀 Starting Physical AI & Humanoid Robotics RAG Backend (Phase 2)...")
    print(f"   Host: {settings.API_HOST}")
    print(f"   Port: {settings.API_PORT}")
    print(f"   Debug: {settings.DEBUG}")
    print(f"   Environment variables loaded: {bool(settings.COHERE_API_KEY)}")

    # Verify required environment variables
    missing_vars = []
    if not settings.COHERE_API_KEY:
        missing_vars.append("COHERE_API_KEY")
    if not settings.QDRANT_URL:
        missing_vars.append("QDRANT_URL")
    if not settings.QDRANT_API_KEY:
        missing_vars.append("QDRANT_API_KEY")
    if not settings.NEON_DATABASE_URL:
        missing_vars.append("NEON_DATABASE_URL")

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        sys.exit(1)

    print("✅ All required environment variables are set")
    print("Starting server...")

    # Start the server
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )


if __name__ == "__main__":
    main()