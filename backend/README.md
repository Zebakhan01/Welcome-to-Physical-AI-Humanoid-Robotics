# AI-Hackathon RAG Chatbot Backend

This is the backend for the RAG (Retrieval-Augmented Generation) chatbot system.

## Tech Stack
- Python
- FastAPI
- LangChain
- Qdrant (local vector store)
- OpenAI-compatible LLM (Qwen / Claude compatible)

## Structure
- `app/main.py` - FastAPI application entry point
- `app/rag/` - RAG modules (loader, embedder, retriever, chat)
- `app/api/` - API endpoints

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload
```