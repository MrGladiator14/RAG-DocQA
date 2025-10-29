# RAG-DocQA - Multi-doc RAG Application with LLMOps

## Overview
A scalable Multi-document Retrieval-Augmented Generation (RAG) application that enables efficient document querying using advanced NLP techniques. This project implements a robust LLMOps pipeline for continuous integration, testing, and deployment.

![chat screen](static/Screenshot%202025-10-29%20002345.png)
![upload screen](static/Screenshot%202025-10-29%20002425.png)

## Features
- Multi-document ingestion and processing
- Vector-based semantic search with MMR
- LLM-powered question answering
- Scalable architecture for production deployment
- Comprehensive testing and CI/CD pipelines

## How it works
- Upload: Files are uploaded to `data/<session_id>/`, split, embedded, and saved as a FAISS index in `faiss_index/<session_id>/`.
- Chat: Each request loads the FAISS index for the given `session_id` and answers using RAG.
- Sessions: A simple in-memory history per session on the server (resets on restart). The browser stores `session_id` in `localStorage`.

## Prerequisites
- Python 3.8+
- Docker & Docker Compose
- AWS Account (for deployment)
- Git

## Quick Start

### Local Development
1. Clone the repository:
   ```bash
   git clone https://github.com/MrGladiator14/RAG-DocQA.git
   cd RAG-DocQA
   ```

2. install uv and Set up a virtual environment:
   ```bash
   uv sync
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Evaluations 🧪

Run LangSmith evaluations on your RAG system:

```bash
python run_evaluations.py

# Run with all evaluators
python run_evaluations.py --evaluator all

# Custom parameters
python run_evaluations.py --evaluator correctness --chunk-size 500 --k 10
```
**Available Evaluators:**
- `correctness` - Custom LLM-as-a-Judge (Gemini 2.5 Pro)
- `cot_qa` - Chain-of-Thought QA evaluator
- `all` - Run all evaluators

## Endpoints
- `GET /` – Serves the UI.
- `GET /health` – Health check.
- `POST /upload` – Form-data file upload. Returns `{ session_id, indexed }`.
- `POST /chat` – JSON body `{ session_id, message }`. Returns `{ answer }`.

## Testing

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=app tests/

# Run specific test file
pytest tests/test_retriever.py -v
```

### Test Structure
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests
- `tests/e2e/` - End-to-end tests

## Docker

### Build and Run
```bash
# Build the Docker image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development with Docker
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up --build
```

## Deployment

### AWS Deployment
1. Github action to deploy to AWS Fargate included

### CI/CD Pipelines

#### GitHub Actions
- Workflows are defined in `.github/workflows/`
- Main workflows:
  - `ci.yml`: Runs on pull requests (linting, testing)
  - `cd.yml`: Deploys to production on merge to main

#### Jenkins
1. Install required plugins:
   - Docker Pipeline
   - AWS Credentials
   - GitHub Integration

2. Create a new Pipeline job and point to `Jenkinsfile`

## Notes
- Ensure your API keys/config are set for the `ModelLoader` to load embeddings/LLM.
- For evaluations, you need `LANGSMITH_API_KEY` and `GOOGLE_API_KEY` in your `.env` file.
- Supported file types: `.pdf`, `.docx`, `.txt`.
- For production, add persistence for chat history and auth; consider cleanup of old session directories.

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
