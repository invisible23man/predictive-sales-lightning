# Base image with Python & Poetry
FROM python:3.10-slim AS base

ENV POETRY_VERSION=1.8.2
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only pyproject + lock for dependency caching
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Copy all code
COPY . .

# Expose ports (for API + Streamlit)
EXPOSE 8000 8501

# CMD is overridden by docker-compose or cloud runtime
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
