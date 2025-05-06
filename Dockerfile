# Base image with Python & Poetry
FROM python:3.10-slim AS base

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Set workdir
WORKDIR /app

# Copy only pyproject + lock first for caching
COPY pyproject.toml poetry.lock ./

# Install deps (no venv)
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction

# Copy rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
