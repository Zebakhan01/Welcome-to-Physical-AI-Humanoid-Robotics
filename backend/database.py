from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL - using environment variable or default
DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("NEON_DATABASE_URL", "sqlite:///./textbook.db"))

# Create engine and session factory (will be initialized when needed)
engine = None
SessionLocal = None

def get_engine():
    global engine, SessionLocal
    if engine is None:
        # Use SQLite as fallback for local development if no database URL is provided
        db_url = os.getenv("DATABASE_URL", os.getenv("NEON_DATABASE_URL", "sqlite:///./textbook.db"))
        if db_url.startswith("postgresql"):
            # For PostgreSQL, create engine with the appropriate settings
            from sqlalchemy import create_engine
            engine = create_engine(db_url, pool_pre_ping=True)
        else:
            # For SQLite, create engine with appropriate settings
            engine = create_engine(db_url, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine

def get_db():
    get_engine()  # Ensure engine is initialized
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base = declarative_base()