# 🧱 Architecture Overview: Predictive Sales Forecasting System

This document explains the high-level system design of the ML pipeline, inference service, dashboard, and MLflow integration.

---

## 🗂️ Component Breakdown

### 1. 🔄 Training Pipeline

- Script: `src/ml/train/train.py`
- Uses `SalesDataModule` to load and normalize data
- Trains CNN + LSTM model
- Saves:
  - Model weights (`model.ckpt`)
  - Normalization stats (`normalization.json`)
- Logs metrics to MLflow

### 2. 🧠 Model: `CNNLSTMForecastModel`

- Conv1D for feature extraction
- LSTM for temporal modeling
- Fully connected layer for regression

### 3. 📈 Evaluation

- Computes RMSE on validation data
- Optionally plots predictions vs. true values
- Metrics & plots logged to MLflow

### 4. 🧠 Inference Module: `SalesForecaster`

- Loads model + normalization
- Normalizes input
- Runs prediction
- Denormalizes output

### 5. 🚀 FastAPI Server

- Endpoint: `/api/predict`
- Validates request (length must match `window_size`)
- Returns forecast as JSON

### 6. 📊 Streamlit Dashboard

- File: `src/ui/streamlit_app/app.py`
- Provides a simple UI to enter input and get predictions
- Can be extended with charts, comparison, evaluation

### 7. 🔧 Configuration (OmegaConf)

- `config/config.yaml` contains:
  - `data` settings (path, window size)
  - `model` architecture
  - `train` params (epochs, paths, mlflow)

---

## 🖼️ System Architecture Diagram

````

```
          +--------------------+
          |  sales_data.csv    |
          +--------------------+
                   |
            [Train Script]
                   |
  +----------------v----------------+
  |     SalesDataModule (norm)      |
  +----------------+----------------+
                   |
       +-----------v----------+
       |  CNN + LSTM Model    | <---+
       +-----------+----------+     |
                   |                |
  +----------------v----------+     |
  |  model.ckpt + norm.json   |-----+
  +---------------------------+
```

\[FastAPI App]             \[Streamlit Dashboard]
\|                            |
+-------v------+            +--------v---------+
\| SalesForecaster|         | Input time series |
\| (Load & predict)|         | via web UI       |
+-------+--------+          +------------------+
|
+-------v--------+
\|  /api/predict  |
+----------------+

```

---

## 🌍 Multi-Service Setup

- **Docker Compose** launches:
  - `api` → FastAPI app
  - `dashboard` → Streamlit UI
  - `mlflow` → Tracking server

---

## 🔮 Possible Enhancements

- Add Prometheus/Grafana for monitoring
- Extend support for multiple item types
- Add MLflow model registry
- Use real time-series validation (e.g., walk-forward)

---

See `README.md` for usage instructions and commands.
