# 🧱 Architecture Overview: Sales Forecasting System

This document outlines the architecture of the ML-powered time series forecasting pipeline including training, inference, APIs, and UI integrations.

---

## 🗂️ Components

### 1. ⚙️ Config System (OmegaConf)
- Centralized `config.yaml` manages:
  - Data source
  - Available categories
  - Model hyperparameters
  - MLflow + API endpoints
- Auto-adjusts based on `MLFLOW_ENV` (e.g., Docker vs local)

---

### 2. 🧠 Training Pipeline

- Modular `train_model_for_category()` trains per category
- Saves:
  - `model_<category>.ckpt`
  - `normalization_<category>.json`
  - Evaluation plot
- Logs all to MLflow with tags (`item_id`, `lr`, etc.)

---

### 3. 🧪 Evaluation

- RMSE, MAE logged per run
- Matplotlib plot saved and logged
- Optionally shown in Streamlit

---

### 4. 🚀 FastAPI Services

| Endpoint        | Purpose                             |
|----------------|-------------------------------------|
| `/api/predict` | Predict sales for given series      |
| `/api/train`   | Train model(s) by category          |
| `/api/train_all` | Train all predefined categories     |

---

### 5. 📊 Streamlit Dashboard

| Feature           | Details                              |
|------------------|--------------------------------------|
| Category Explorer| Line plots + rolling average         |
| Predict Sales    | Manual input → model → prediction    |
| Train Models     | Select categories, run training live |

All routes dynamically load categories from `config.yaml`.

---

## 🔄 MLflow Integration

- Experiment: `sales-forecast`
- Run name: `CNNLSTM-{category}`
- Metrics, plots, checkpoints, and configs logged
- Uses local SQLite backend by default (Docker volume)

---

## 🧱 System Diagram

````

```
               +-------------+
               | sales_data.csv
               +-------------+
                      |
               [Train Script / FastAPI]
                      |
         +------------v-------------+
         | SalesDataModule (norm)   |
         +------------+-------------+
                      |
              +-------v--------+
              | CNNLSTM Model  |
              +-------+--------+
                      |
           +----------v--------+           +--------------+
           | model_<cat>.ckpt  |---------> | FastAPI /predict |
           | norm_<cat>.json   |           +--------------+
           +-------------------+                |
                                                v
                                      +-------------------+
                                      | Streamlit Forecast |
                                      +-------------------+
```

```

---

## 🚀 CI + Reproducibility

- `Dockerfile` + `docker-compose.yml`
- `poetry.lock` for pinned dependencies
- Configurable training & inference
- MLflow for all experiments
- Test coverage via Pytest
- Lint: `black`, `isort`, `flake8` via Pre-commit

---

## 🌱 Future Enhancements

- Support multi-step forecasting
- Add Prometheus + Grafana monitoring
- Use LangChain or RAG for user-guided analytics
- Scale with Kubernetes

---

📁 See `README.md` for usage guide.
```
