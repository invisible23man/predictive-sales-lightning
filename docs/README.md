# 📦 Predictive Sales Forecasting App (Lightning + FastAPI)

A production-grade machine learning application that forecasts sales using a CNN-LSTM model and serves predictions via a FastAPI REST API. The app is containerized with Docker and follows best practices including config-driven training, reproducibility, and CI/CD readiness.

---

## 🚀 Usage Guide

### 🔧 Requirements

* Docker + Docker Compose
* Python 3.10 (for local development)

---

### 🐳 Run via Docker Compose

```bash
docker-compose up --build
```

Open docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🔁 Train the Model (Locally)

```bash
poetry run python -m src.ml.train.train
```

This will:

* Train a CNN-LSTM time series model
* Save the model checkpoint to `checkpoints/model.ckpt`
* Save normalization stats to `checkpoints/normalization.json`

---

### 🌐 API Endpoint

**POST /api/predict**

**Request Body:**

```json
{
  "series": [100.0, 120.5, 130.2, ...]  // must match `window_size` (e.g., 14)
}
```

**Response:**

```json
{
  "predicted_sales": 148.32
}
```

---

### 🧪 Tests

```bash
poetry run pytest
```

---

### ✅ Pre-commit Hooks (Linting)

```bash
poetry run pre-commit run --all-files
```

Includes: `black`, `isort`, `flake8`

---

### 📁 Project Structure (Simplified)

```
├── src
│   ├── app                  # FastAPI app
│   │   ├── api
│   │   └── services
│   ├── ml                   # ML logic
│   │   ├── data
│   │   ├── models
│   │   ├── train
│   └── config               # Hydra/OmegaConf setup
├── checkpoints              # Saved models and stats
├── tests                    # TDD-based test coverage
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── config/config.yaml
```

---

### 📦 ML Pipeline Summary

* CNN + LSTM for sequential sales prediction
* Sliding window time series data module
* PyTorch Lightning for clean training loop
* Model & stats checkpointing
* MLflow logging supported (optional)

---

## ✍️ Author

Gowrisankar — MLOps & ML Engineer | Pythonista | Cloud-native Systems

---

See `docs/architecture.md` for system design and code architecture diagram.
