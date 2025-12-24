# Quickstart Guide: RAG Chatbot Backend

## Overview
This guide provides instructions for setting up, configuring, and running the RAG Chatbot Backend for the Physical AI & Humanoid Robotics textbook.

## Prerequisites
- Python 3.11 or higher
- Access to Cohere API (API key)
- Access to Qdrant Cloud (API key and URL)
- Access to Neon Serverless PostgreSQL (connection string)

## Setup Instructions

### 1. Clone and Navigate to Backend Directory
```bash
# If starting fresh, create the backend directory
mkdir -p backend
cd backend
```

### 2. Create Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip

# Install dependencies based on requirements.txt that will be created
pip install fastapi uvicorn cohere qdrant-client sqlalchemy psycopg2-binary python-dotenv pydantic
```

### 3. Environment Configuration
Create a `.env` file in the backend root with the following variables:

```env
# Cohere Configuration
COHERE_API_KEY=your_cohere_api_key_here

# Qdrant Cloud Configuration
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_HOST=your_qdrant_cloud_url_here
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=textbook_content

# Neon PostgreSQL Configuration
DATABASE_URL=postgresql://username:password@ep-xxxxxx-pooler.us-east-1.aws.neon.tech/dbname?sslmode=require

# Application Configuration
APP_ENV=development
LOG_LEVEL=info
```

### 4. Directory Structure Setup
Create the following directory structure in your backend folder:

```
backend/
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   │   ├── routers/
│   │   └── dependencies.py
│   └── config/
├── tests/
├── requirements.txt
└── main.py
```

### 5. Initialize the Database
The application will automatically create required tables on startup if they don't exist. Ensure your Neon PostgreSQL instance is accessible via the DATABASE_URL.

### 6. Run the Application
```bash
cd backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

## Basic Usage

### 1. Check API Health
```bash
curl http://localhost:8000/api/v1/health
```

### 2. Index Textbook Content
First, ensure your Docusaurus textbook content is available in a local directory, then trigger indexing:

```bash
curl -X POST http://localhost:8000/api/v1/content/reindex
```

### 3. Ask a Question
```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main components of a humanoid robot?"
  }'
```

### 4. Ask About Selected Text
```bash
curl -X POST http://localhost:8000/api/v1/rag/ask-selected \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does this section say about actuators?",
    "selected_text": "Humanoid robots use various types of actuators including servo motors, hydraulic systems, and pneumatic systems to achieve movement..."
  }'
```

## Configuration Options

### Environment Variables
- `APP_ENV`: Set to "production" for production environments (default: "development")
- `LOG_LEVEL`: Set logging level (default: "info", options: debug, info, warning, error)
- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: "textbook_content")
- `MAX_CONTENT_CHUNK_SIZE`: Maximum size of content chunks in characters (default: 1000)
- `VECTOR_DIMENSION`: Dimension of the Cohere embeddings (default: 1024 for multilingual v3)

## API Endpoints

### RAG Endpoints
- `POST /api/v1/rag/ask` - General textbook question answering
- `POST /api/v1/rag/ask-selected` - Selected-text question answering

### Content Management
- `GET /api/v1/content/status` - Get indexing status
- `POST /api/v1/content/reindex` - Trigger content re-indexing

### System
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/docs` - API documentation (Swagger UI)

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify all API keys and connection strings are correct
   - Check network connectivity to Cohere, Qdrant Cloud, and Neon PostgreSQL

2. **Indexing Failures**
   - Ensure the Docusaurus content directory is accessible
   - Check that Cohere API is responding
   - Verify Qdrant Cloud collection exists and is writable

3. **Slow Response Times**
   - Check if the content is properly indexed
   - Verify vector search is working correctly
   - Monitor database and vector store performance

### Validation Commands
```bash
# Test Cohere connection
python -c "import cohere; co = cohere.Client('your_key_here'); print(co.embed(texts=['test'], model='embed-multilingual-v3.0'))"

# Test Qdrant connection
python -c "from qdrant_client import QdrantClient; qdrant_client = QdrantClient(url='your_url_here', api_key='your_key_here'); print(qdrant_client.get_collections())"

# Test PostgreSQL connection
python -c "import psycopg2; conn = psycopg2.connect('your_db_url_here'); print('Connected successfully')"
```

## Next Steps
1. Configure your Docusaurus textbook content directory
2. Run the initial content indexing
3. Test the question answering functionality
4. Validate response accuracy and citations