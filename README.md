# Predictive Maintenance API for Turbofan Engine Sensor Data

End-to-end machine learning system for predicting **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor time-series data. Built to demonstrate the complete production ML loop: data ingestion → feature engineering → experiment tracking → model registry → containerised REST API → monitoring dashboard.

---

## Results

Evaluated on the NASA CMAPSS FD001 test set (100 engines, completely held out during training and hyperparameter search):

| Model | RMSE (cycles) | MAE (cycles) | CMAPSS Score |
|---|---|---|---|
| Random Forest | 19.6 | 14.4 | 1,142 |
| XGBoost | 19.7 | 14.8 | 1,199 |
| **LSTM (tuned)** | **21.2** | **16.0** | **1,093** |

The CMAPSS scoring function penalises late predictions (overestimating RUL) exponentially more than early ones — reflecting the real asymmetric cost of unexpected failure vs unnecessary maintenance.

This is why the LSTM is the selected model despite having the *highest* RMSE of the three: it posts the lowest CMAPSS score — **4.2% below the Random Forest and 8.8% below XGBoost** — i.e. fewer of the overestimates that let an engine run past its failure point. RMSE treats both error directions equally; CMAPSS does not, and neither does a maintenance budget.

Hyperparameters were selected via **Optuna Bayesian search** (20 trials, each logged as a nested MLflow run).

---

## Stack

| Layer | Technology |
|---|---|
| ETL orchestration | Prefect |
| Feature store | DuckDB |
| Feature engineering | Custom sklearn-style `FeatureEngineer` class |
| Baseline models | scikit-learn, XGBoost |
| Deep learning | PyTorch (LSTM) |
| Hyperparameter search | Optuna |
| Experiment tracking | MLflow Tracking (SQLite backend) |
| Model registry | MLflow Model Registry |
| API framework | FastAPI + Uvicorn |
| Validation | Pydantic |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
predictive-maintenance/
├── data/raw/               # NASA CMAPSS dataset (not tracked in git)
├── database/               # DuckDB feature store
├── artifacts/              # Fitted FeatureEngineer, scaler, model weights
├── src/
│   ├── pipeline/
│   │   └── etl_flow.py     # Prefect ETL flow (extract → transform → load)
│   ├── features/
│   │   └── feature_engineering.py   # FeatureEngineer class
│   ├── training/
│   │   ├── models.py       # RULPredictor (LSTM) architecture
│   │   ├── utils.py        # Shared utilities (sequences, metrics, scoring)
│   │   ├── train_baselines.py  # Random Forest + XGBoost training
│   │   ├── train_lstm.py   # LSTM + Optuna hyperparameter search
│   │   └── evaluate.py     # Final test set evaluation + model registration
│   ├── api/
│   │   └── main.py         # FastAPI application
│   └── dashboard/
│       ├── app.py          # Streamlit dashboard (fleet / engine / model views)
│       └── predictions.csv # Test-set predictions, exported from eda.ipynb
├── notebooks/
│   └── eda.ipynb           # Exploratory analysis + predictions.csv export
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── mlflow.db               # MLflow tracking backend
├── requirements.txt
└── requirements-dashboard.txt   # separate env, see below
```

---

## Dataset

[NASA CMAPSS Turbofan Engine Degradation Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — a standard benchmark for predictive maintenance research. FD001 subset: 100 training engines, 100 test engines, 21 sensor channels, single fault mode (HPC degradation), single operating condition.

RUL labels are computed from run-to-failure trajectories and capped at 125 cycles — sensors show no degradation signal before this point, so early lifecycle predictions are not meaningful.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The dashboard needs a **separate environment**:

```bash
pip install -r requirements-dashboard.txt
```

Streamlit requires `protobuf>=3.20`, while the pinned `tensorflow==2.10.1` in
`requirements.txt` requires `protobuf<3.20`. The two cannot coexist, so the
dashboard dependencies are kept in their own file.

### 2. Download the dataset

Download [CMAPSS Dataset 6](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) and place the contents in `data/raw/`.

### 3. Run the ETL pipeline

Extracts raw sensor data, engineers rolling features, and loads into DuckDB. Also fits and saves the `FeatureEngineer` artifact.

```bash
python -m src.pipeline.etl_flow
```

### 4. Train baseline models

```bash
python -m src.training.train_baselines
```

### 5. LSTM hyperparameter search

Runs 20 Optuna trials, each logged as a nested MLflow run.

```bash
python -m src.training.train_lstm
```

### 6. Final test set evaluation

Trains on all data with the best hyperparameters, evaluates on the held-out test set, and registers the model in the MLflow registry.

```bash
python -m src.training.evaluate
```

### 7. View experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` to compare all runs.

### 8. Run the API locally

```bash
uvicorn src.api.main:app --reload
```

Interactive docs available at `http://127.0.0.1:8000/docs`.

### 9. Run the API in Docker

```bash
cd docker
docker-compose up --build
```

### 10. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501` with three views: **Fleet Overview** (all 100
test engines by predicted RUL, click a bar to drill into an engine), **Engine
Inspector** (sensor trajectories for a single unit), and **Model Performance**
(metrics, model comparison, error breakdown, worst predictions).

Reads `src/dashboard/predictions.csv`; regenerate it from `notebooks/eda.ipynb`
after retraining.

---

## API

### `POST /predict`

Send a pre-engineered feature window (30 timesteps × 45 features) and receive a RUL prediction.

**Request body:**
```json
{
  "sensor_readings": [[...], [...], ...]
}
```

**Response:**
```json
{
  "predicted_rul": 47.3,
  "failure_probability": 0.39
}
```

`failure_probability` is derived from predicted RUL via exponential decay: `exp(-RUL / 50)`.

### `GET /health`

Returns model status and configuration.

---

## Feature Engineering

Raw sensor signals are noisy with degradation manifesting as slow trend changes rather than sharp value shifts. Three feature types are computed per sensor over a rolling window (w=25 cycles):

- **Rolling mean** — smooths noise, captures level
- **Rolling std** — captures increasing volatility near failure
- **First difference** — captures rate of change / slope

Low-variance sensors (constant across all cycles) are identified on training data and excluded. The fitted `FeatureEngineer` is serialised and reused identically at inference — no statistics are recomputed on new data.

---

## Design Notes

**Why DuckDB?** Processed features are stored in a local SQL database rather than flat files. This supports subset queries without loading the full dataset into memory and mirrors the warehouse layer in production data stacks.

**Why Prefect?** The ETL steps (extract → transform → load) are structured as a Prefect flow. In production this would be scheduled to run as new sensor data arrives, triggering retraining automatically.

**Why GroupKFold?** Train/validation splits are made at the engine level, not the row level. Splitting rows randomly would leak autocorrelated data across the boundary, producing optimistic validation metrics.

**Docker note:** Local deployment loads model weights directly from `artifacts/best_model.pt`. In a cloud deployment the API would load from a remote MLflow tracking server or S3 artifact store, avoiding the SQLite filesystem limitations present in the local setup.
