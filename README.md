# Titanic MLOps App

> This repo showcases an implementation of MLOps principles on the Titanic dataset. It features a prediction service and tools for monitoring model performance and data drift.

![Demo video](https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/intro-mlops/demo-titanic-mlops.mp4)

## Architecture

```
+-----------------------------------------------------+
|                   Docker Compose                     |
|                                                      |
|  +----------+   +----------+   +----------------+   |
|  |  Nginx   |-->| FastAPI  |-->|  MLflow Server |   |
|  |  (HTTPS) |   |  (API)   |   |  (tracking)    |   |
|  +----------+   +----------+   +-------+--------+   |
|       |              |                 |             |
|       |              |         +-------+--------+   |
|       v              v         |   PostgreSQL   |   |
|  +----------+   +----------+  |   (metadata)   |   |
|  |Monitoring|   |  MinIO   |  +----------------+   |
|  |(Evidently)|  |(artifacts)|                       |
|  +----------+   +----------+                        |
+-----------------------------------------------------+
         Dataset versioned with DVC + DagsHub
```

## Service URLs

Once the stack is running, the following services are available:

| Service          | URL                              | Description                          |
|------------------|----------------------------------|--------------------------------------|
| **Frontend**     | https://localhost                 | Web UI for predictions and MLOps     |
| **API Docs**     | https://localhost/api/docs        | Swagger / OpenAPI documentation      |
| **MLflow**       | http://localhost:5001             | Experiment tracking and model registry (click "Experiments" in the sidebar) |
| **MinIO**        | http://localhost:9001             | S3-compatible artifact store console |
| **Monitoring**   | https://localhost/monitoring/     | Evidently drift reports              |

> **Note**: HTTPS uses self-signed certificates. Accept the browser warning or use `curl -k`.

## Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **make** (optional but recommended for convenience commands)

### Installing make

**macOS** (comes pre-installed with Xcode Command Line Tools):

```bash
# If make is not available, install the command line tools:
xcode-select --install
```

**Ubuntu / Debian**:

```bash
sudo apt-get update && sudo apt-get install -y make
```

**Fedora / RHEL**:

```bash
sudo dnf install make
```

**Windows** (via Chocolatey):

```bash
choco install make
```

Alternatively, you can skip make entirely and run `docker compose up --build -d` directly.

## Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/Lalanne0/titanic-mlops-app.git
cd titanic-mlops-app

# 2. Copy the example env file and fill in your passwords
cp .env.example .env

# 3. Start everything (builds and starts all containers)
make up

# 4. Open the frontend at https://localhost
#    Train a model from the UI, then start making predictions.
```

## API Endpoints

| Method | Endpoint              | Description                                  |
|--------|-----------------------|----------------------------------------------|
| GET    | `/api/health`         | Health check                                 |
| POST   | `/api/predict`        | Predict survival for a passenger             |
| POST   | `/api/feedback`       | Submit a corrected label ("prediction was wrong") |
| POST   | `/api/retrain`        | Train or retrain the model on the current dataset |
| POST   | `/api/simulate-drift` | Inject drifted data and generate a drift report |
| POST   | `/api/reset-data`     | Restore the original dataset                 |
| GET    | `/api/model-info`     | Current model version and metadata           |
| GET    | `/api/dataset-info`   | Dataset statistics                           |

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
| `make up`          | Build and start all services               |
| `make down`        | Stop all services                          |
| `make restart`     | Rebuild and restart                        |
| `make logs`        | Follow all container logs                  |
| `make clean`       | Remove containers, volumes, and images     |
| `make health`      | Check API health                           |
| `make retrain`     | Train or retrain the model                 |
| `make predict`     | Run a sample prediction                    |
| `make drift`       | Simulate data drift (500 samples)          |
| `make reset-data`  | Restore original dataset                   |
| `make dvc-init`    | Initialize DVC with DagsHub remote         |
| `make dvc-push`    | Push dataset to DagsHub                    |

## DVC Dataset Versioning

```bash
# Set your DagsHub token first
export DAGSHUB_TOKEN=<your-dagshub-token>

# Initialize DVC + DagsHub remote
make dvc-init

# Push dataset to remote
make dvc-push

# Pull dataset (on another machine)
make dvc-pull
```

## MLOps Workflow

1. **Train** -- `POST /api/retrain` logs metrics, parameters, and the model to MLflow
2. **Predict** -- `POST /api/predict` uses the latest model from the MLflow registry
3. **Feedback** -- `POST /api/feedback` adds the corrected label to the working dataset
4. **Retrain** -- `POST /api/retrain` trains a new model version on the updated dataset
5. **Drift** -- `POST /api/simulate-drift` injects drifted data and generates an Evidently report (no auto-retrain)
6. **Monitor** -- Open the Evidently dashboard or the generated report to inspect detected drift
7. **Reset** -- `POST /api/reset-data` restores the dataset to its original state

## Environment Variables

See [`.env.example`](.env.example) for all required configuration.

## Tech Stack

- **API**: FastAPI + scikit-learn + Uvicorn
- **Tracking**: MLflow + PostgreSQL + MinIO
- **Monitoring**: Evidently AI
- **Proxy**: Nginx (self-signed HTTPS)
- **Data versioning**: DVC + DagsHub
- **Orchestration**: Docker Compose