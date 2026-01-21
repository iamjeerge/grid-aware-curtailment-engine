# Backend Dockerfile for Grid-Aware Curtailment Engine
FROM python:3.11-slim

# Install system dependencies including GLPK solver and curl for healthcheck
RUN apt-get update && apt-get install -y \
    glpk-utils \
    libglpk-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual env (we're in container)
RUN poetry config virtualenvs.create false

# Install dependencies (production only, skip project itself)
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
