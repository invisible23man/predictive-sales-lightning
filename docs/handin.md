# 📝 Take-Home Assignment: Predictive Sales Model & API Deployment

**Developer:** Gowrisankar

---

## ✅ Assignment Overview

Build a production-ready machine learning application that:

1. Trains a model to forecast future sales
2. Serves predictions via a REST API
3. Is containerized with Docker and easy to run

---

## 🧠 Modelling Approach

### ➤ Model Architecture

* **CNN-LSTM** hybrid:

  * CNN: temporal pattern extraction
  * LSTM: sequence learning
  * FC Layer: regression output
* Implemented using **PyTorch Lightning**

### ➤ Prediction Target

* Forecast **next-day total sales** per `Product Category`
* Each model is trained independently per category (`Beauty`, `Clothing`, etc.)

### ➤ Preprocessing

* Daily aggregation
* Resampling with forward-fill
* Sliding window generation (configurable window size)
* Normalization: mean + std saved per category

### ➤ Assumptions

* Only historical sales (`Date`, `Product Category`, `Total Amount`) are available
* Forecast horizon = 1 day ahead
* No additional covariates (e.g., promotions, store info)

---

## 🌐 API Development

### ➤ Built with FastAPI

#### `/api/predict` (POST)

* Input: last `N` sales (N = window size)
* Output: predicted next-day sales value

#### `/api/train` (POST)

* Input: list of categories + experiment name
* Triggers training via ML pipeline

#### `/api/train_all` (POST)

* Trains models for **all available categories** (loaded from config)

---

## 📊 Streamlit Dashboard

### Pages

* **Data Explorer:** historical sales, rolling mean, per-category
* **Predict Sales:** manual input + forecast result
* **Train Models:** trigger training per category or all at once

---

## 🐳 Containerization

### ➤ Tools Used

* `Dockerfile`: Python 3.10, Poetry, FastAPI, Streamlit
* `docker-compose.yml`: spins up

  * FastAPI app
  * Streamlit dashboard
  * MLflow tracking server

### ➤ One-Liner Setup

```bash
docker compose up --build
```

---

## ✅ Deliverables

* ✅ `src/` with clean modular code (`ml/`, `app/`, `ui/`)
* ✅ `Dockerfile` + `docker-compose.yml`
* ✅ `README.md` with:

  * Setup instructions
  * API usage and curl examples
  * Modelling approach
* ✅ `docs/architecture.md`: system design
* ✅ `tests/`: unit test coverage (data module, API health, etc.)
* ✅ `checkpoints/`: saved models + normalization stats
* ✅ MLflow logs + visual artifacts

---

## 🧪 Evaluation Summary

| Criteria            | Implementation                                   |
| ------------------- | ------------------------------------------------ |
| ✅ Correct model     | CNN-LSTM, per category                           |
| ✅ Code quality      | Clean, typed, modular, tested                    |
| ✅ API design        | FastAPI + JSON schemas                           |
| ✅ Dockerization     | One command setup, container-per-service         |
| ✅ Thoughtful design | Configurable, reusable pipeline                  |
| ✅ CI-Ready          | Pre-commit hooks, testable components            |
| ✅ Optional Features | Streamlit UI, MLflow logging, config abstraction |

---

## 🚀 Production Enhancements (Planned / Suggested)

* Add caching layer (e.g., Redis) for predictions
* Store trained models in a registry (e.g., S3 or GCS)
* Add Prometheus/Grafana monitoring
* Use Airflow for scheduled retraining
* Kubernetes/Cloud Run deployment ready

---

## 🙌 Final Notes

* The entire system was developed with **clarity, maintainability, and testability** in mind.
* Fast iteration and training across multiple product categories is supported out-of-the-box.
* Configurations (model, paths, categories, infra) are entirely centralized and Docker-aware.
