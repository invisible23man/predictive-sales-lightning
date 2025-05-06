# 📦 Predictive Sales Forecasting App (Lightning + FastAPI + Streamlit)

A production-grade machine learning application that forecasts sales using a CNN-LSTM model and serves predictions via:

- 🚀 FastAPI REST API
- 📊 Streamlit Dashboard
- 📁 MLflow Tracking UI

The entire stack is containerized with Docker Compose and built with reproducibility, modularity, and MLOps best practices in mind.

---

## 🚀 Usage Guide

### 🔧 Requirements

- Docker + Docker Compose
- Poetry (for local development)
- Python 3.10+

---

### 🐳 Run the Full Stack

```bash
docker compose up --build
````

Services:

* FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
* Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)
* MLflow Tracking UI: [http://localhost:5000](http://localhost:5000)

---

### 🔁 Train the Model (Locally)

```bash
poetry run python -m src.ml.train.train
```

This will:

* Train a CNN-LSTM time series model
* Save the model checkpoint to `checkpoints/model_<category>.ckpt`
* Save normalization stats to `checkpoints/normalization_<category>.json`
* Log training metrics and artifacts to MLflow

---

### 📡 API Endpoints

#### 🔹 Predict

**POST** `/api/predict`

```json
Request:
{
  "series": [105.5, 110.2, ..., 99.9]
}

Response:
{
  "predicted_sales": 148.32
}
```

#### 🔹 Train

**POST** `/api/train`

```json
{
  "categories": ["Beauty", "Clothing"],
  "experiment_name": "sales-forecast"
}
```

**POST** `/api/train_all`

Triggers training for all predefined categories (loaded from config).

---

### 📊 Dashboard (Streamlit)

* Explore historical sales trends per category
* Trigger model training per category or in bulk
* Enter series manually and get predictions
* Visualize rolling averages and evaluation plots

---

## 🧰 Dev Commands

```bash
# Lint & formatting
poetry run pre-commit run --all-files

# Run tests
poetry run pytest

# Train manually
poetry run python -m src.ml.train.train
```

---

## 📁 Project Structure (Simplified)

```
.
├── src
│   ├── app/               # FastAPI routes & services
│   ├── config/            # YAML config + loader
│   ├── ml/                # Training pipeline, models
│   └── ui/                # Streamlit app
├── checkpoints/           # Saved models + stats
├── mlruns/                # MLflow logs
├── data/                  # Raw input
├── tests/                 # Unit tests
├── docs/                  # Architecture docs
├── docker-compose.yml
└── pyproject.toml
```

---

## ✍️ Author

Gowrisankar — MLOps Engineer

See `docs/architecture.md` for the full design walkthrough.
See `docs/handin.md` for the assignment related additional documentation.
