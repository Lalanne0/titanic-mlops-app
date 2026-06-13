# Titanic MLOps App

> **Would you survive the Titanic?** — An MLOps-powered prediction service showcasing the full lifecycle of a machine learning project in production.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Nginx   │──▸│ FastAPI  │──▸│  MLflow Server │  │
│  │  (HTTPS) │   │  (API)   │   │  (tracking)    │  │
│  └──────────┘   └──────────┘   └───────┬────────┘  │
│       │              │                  │            │
│       │              │          ┌───────┴────────┐  │
│       ▼              ▼          │   PostgreSQL   │  │
│  ┌──────────┐   ┌──────────┐   │   (metadata)   │  │
│  │Monitoring│   │  MinIO   │   └────────────────┘  │
│  │(Evidently)│  │(artifacts)│                       │
│  └──────────┘   └──────────┘                        │
└─────────────────────────────────────────────────────┘
         Dataset versioned with DVC + DagsHub
```

| Service        | Port  | Description                           |
|----------------|-------|---------------------------------------|
| **Nginx**      | 443   | HTTPS reverse proxy                   |
| **FastAPI**    | 8000  | Prediction API (via `/api/`)          |
| **MLflow**     | 5001  | Experiment tracking & model registry  |
| **PostgreSQL** | 5432  | MLflow metadata backend               |
| **MinIO**      | 9001  | S3-compatible artifact store          |
| **Monitoring** | 8080  | Evidently drift reports (via `/monitoring/`) |

## Quick Start

```bash
# 1. Clone & navigate
git clone https://github.com/Lalanne0/titanic-mlops-app.git
cd titanic-mlops-app

# 2. Start everything (builds + starts 7 containers)
make up

# 3. Wait ~30s for auto-training, then open:
#    API Docs:   https://localhost/api/docs
#    MLflow UI:  http://localhost:5001
#    Monitoring: https://localhost/monitoring/
#    MinIO:      http://localhost:9001
```

> **Warning**: HTTPS uses self-signed certificates — accept the browser warning or use `curl -k`.

## API Endpoints

| Method | Endpoint            | Description                                  |
|--------|---------------------|----------------------------------------------|
| GET    | `/api/health`       | Health check                                 |
| POST   | `/api/predict`      | Predict survival for a passenger             |
| POST   | `/api/feedback`     | "Got it wrong?" — submit corrected label     |
| POST   | `/api/retrain`      | Retrain model on current dataset             |
| POST   | `/api/simulate-drift` | Inject drifted data + auto-retrain         |
| GET    | `/api/model-info`   | Current model version & metadata             |
| GET    | `/api/dataset-info` | Dataset statistics                           |

### Example: Predict

```bash
curl -sk -X POST https://localhost/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100,
    "Embarked": "S"
  }'
```

## Make Commands

| Command            | Description                                |
|--------------------|--------------------------------------------|
| `make up`          | Build & start all services                 |
| `make down`        | Stop all services                          |
| `make restart`     | Rebuild & restart                          |
| `make logs`        | Follow all container logs                  |
| `make clean`       | Remove containers, volumes & images        |
| `make health`      | Check API health                           |
| `make retrain`     | Trigger model retraining                   |
| `make predict`     | Run a sample prediction                    |
| `make drift`       | Simulate data drift (100 samples)          |
| `make dvc-init`    | Initialize DVC with DagsHub remote         |
| `make dvc-push`    | Push dataset to DagsHub                    |

## DVC Dataset Versioning

```bash
# Initialize DVC + DagsHub remote
make dvc-init

# Push dataset to remote
make dvc-push

# Pull dataset (on another machine)
make dvc-pull
```

## MLOps Workflow

1. **Train** → `POST /api/retrain` → logs metrics, params, model to MLflow
2. **Predict** → `POST /api/predict` → uses latest model from MLflow registry
3. **Feedback** → `POST /api/feedback` → adds corrected data to dataset
4. **Retrain** → `POST /api/retrain` → new model version registered
5. **Drift** → `POST /api/simulate-drift` → injects drifted data, retrains, generates Evidently report
6. **Monitor** → `GET /monitoring/` → view drift detection reports

## Environment Variables

See [`.env.example`](.env.example) for all required configuration.

## Tech Stack

- **API**: FastAPI + scikit-learn + Uvicorn
- **Tracking**: MLflow + PostgreSQL + MinIO
- **Monitoring**: Evidently AI
- **Proxy**: Nginx (self-signed HTTPS)
- **Data versioning**: DVC + DagsHub
- **Orchestration**: Docker Compose