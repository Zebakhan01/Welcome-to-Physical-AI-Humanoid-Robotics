# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js (v16 or higher)
- Python (v3.9 or higher)
- Git
- Access to OpenAI API key
- Access to Qdrant Cloud account
- Access to Neon Serverless Postgres account

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd physical-ai-textbook
```

### 2. Frontend Setup (Docusaurus)
```bash
# Navigate to project root
cd physical-ai-textbook

# Install frontend dependencies
npm install

# Create environment file
cp .env.example .env

# Edit .env with your frontend configuration
```

### 3. Backend Setup (FastAPI)
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env with your backend configuration
```

### 4. Environment Configuration

Create `.env` files for both frontend and backend with the following variables:

**Frontend (.env):**
```
# Frontend Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_SITE_URL=http://localhost:3000
```

**Backend (.env):**
```
# API Configuration
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=your_neon_database_url
SECRET_KEY=your_secret_key_for_auth
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Running the Application

### 1. Start the Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload
```
The backend will be available at `http://localhost:8000`

### 2. Start the Frontend (Docusaurus)
```bash
cd physical-ai-textbook  # project root
npm start
```
The frontend will be available at `http://localhost:3000`

## Project Structure

```
physical-ai-textbook/
├── docs/                # Textbook content (Markdown files)
│   ├── intro/           # Introduction chapters
│   ├── weeks/           # Week-by-week content (1-13)
│   ├── modules/         # Specialized modules (ROS, Gazebo, etc.)
│   ├── capstone/        # Capstone project content
│   ├── hardware/        # Hardware guide
│   └── appendix/        # Reference materials
├── src/                 # Docusaurus custom components
│   ├── components/      # React components
│   ├── pages/           # Custom pages
│   └── css/             # Custom styles
├── backend/             # FastAPI backend
│   ├── api/             # API routes
│   ├── models/          # Data models
│   ├── utils/           # Utility functions
│   ├── main.py          # Application entry point
│   └── requirements.txt # Python dependencies
├── static/              # Static assets (images, diagrams)
├── docusaurus.config.js # Docusaurus configuration
├── package.json         # Frontend dependencies
└── README.md            # Project overview
```

## Key Features

### 1. Textbook Navigation
- Organized by weeks (1-13) and specialized modules
- Clear learning objectives for each chapter
- Cross-references between related topics

### 2. RAG Chatbot
- Ask questions about textbook content
- Context-aware responses based on current chapter
- Selected-text questioning capability

### 3. Content Structure
- Beginner-friendly explanations with technical depth
- Code examples and practical exercises
- Diagrams and visual aids throughout

## Development Workflow

### Adding New Content
1. Create new Markdown files in appropriate `docs/` subdirectory
2. Update `sidebars.js` to include new content in navigation
3. Add learning objectives and prerequisites as frontmatter

### Adding Backend Features
1. Create new route in appropriate API module
2. Define Pydantic models for request/response validation
3. Implement business logic in service modules
4. Add tests for new functionality

### Testing
```bash
# Frontend tests
npm test

# Backend tests
cd backend
python -m pytest tests/
```

## Deployment

### Frontend (GitHub Pages)
```bash
npm run build
npm run deploy
```

### Backend (Cloud Platform)
Deploy the `backend/` directory to your preferred cloud platform (Railway, Render, etc.) with the required environment variables.

## Troubleshooting

### Common Issues

1. **Port already in use**: Change ports in package.json or kill existing processes
2. **API connection errors**: Verify backend is running and environment variables are set
3. **Content not showing**: Check file paths and sidebar configuration

### Getting Help
- Check the documentation in `docs/` for detailed information
- Review the API documentation at `/docs` when backend is running
- Look at existing examples for patterns to follow