# Titanic MLOps App

> **Would you survive the Titanic?** -- An MLOps-powered prediction service showcasing the full lifecycle of a machine learning project in production, with an optional LLMOps interaction layer.

<video controls>
    <source src="./demo/Titanic%20MLOps%20-%20short%20demo.mp4" type="video/mp4">
</video>

## Architecture

```
+--------------------------------------------------------------+
|                       Docker Compose                          |
|                                                               |
|  +-----------+    +------------------+    +----------------+  |
|  |   Nginx   |--->|     FastAPI      |--->|  MLflow Server |  |
|  |  (HTTPS)  |    |  (API + Chat)    |    |  (tracking)    |  |
|  +-----------+    +--------+---------+    +-------+--------+  |
|       |                    |                      |           |
|       |           +--------+--------+     +-------+--------+  |
|       v           |                 |     |   PostgreSQL   |  |
|  +-----------+    |  LLM Provider   |     |   (metadata)   |  |
|  | Frontend  |    |  (optional,     |     +----------------+  |
|  | (SPA)     |    |   via API key)  |                         |
|  +-----------+    +-----------------+     +----------------+  |
|       |                                   |     MinIO      |  |
|  +-----------+                            |  (artifacts)   |  |
|  |Monitoring |                            +----------------+  |
|  |(Evidently)|                                                |
|  +-----------+                                                |
+--------------------------------------------------------------+
             Dataset versioned with DVC + DagsHub
```

**Two operational loops in one demo:**

```
MLOps:   data / model quality -> drift -> feedback -> retraining
LLMOps:  prompt / tool use / context quality -> traces and evaluation -> safer responses
```

## Service URLs

Once the stack is running, the following services are available:

| Service          | URL                              | Description                          |
|------------------|----------------------------------|--------------------------------------|
| **Frontend**     | https://localhost                 | Web UI for predictions, Copilot, and MLOps |
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
| POST   | `/api/chat`           | Send a message to the Copilot (LLMOps layer) |
| GET    | `/api/traces`         | View recent LLM interaction traces           |

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

### Example: Chat (Copilot)

```bash
curl -sk -X POST https://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Would a 29-year-old woman in first class survive?"}'
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
| `make dvc-push`    | Push dataset to DagsHub (needs a token)    |
| `make dvc-pull`    | Pull dataset from DagsHub (needs a token)  |

## MLOps vs LLMOps in This Demo

This application demonstrates two independent operational loops on the same product:

**MLOps loop** (the existing classifier):
- Train a logistic regression on Titanic data, versioned in MLflow
- Serve predictions through a deterministic API endpoint
- Monitor for data drift with Evidently
- Collect user feedback and retrain on updated data
- Track every training run's metrics, parameters, and model artifacts

**LLMOps loop** (the Copilot layer):
- A language model turns natural-language questions into structured tool calls to the prediction API
- The LLM never generates predictions itself. It calls `predict_survival` as a tool, keeping the sklearn model as the source of truth
- A model card provides grounding context so the LLM communicates limitations accurately
- Each interaction is traced: prompt, model identifier, tool calls, latency, token usage, and estimated cost
- An evaluation set with representative cases tests correct tool invocation, grounded predictions, and safety (no causal claims, no fabricated numbers)

The key difference: MLOps optimizes **model quality** (accuracy, drift, retraining). LLMOps optimizes **interaction quality** (correct tool use, grounded answers, safe responses, cost efficiency).

### Copilot Configuration

The Copilot uses the OpenAI-compatible API format. By default it points to [Groq](https://groq.com/) (free tier available). Set these in your `.env`:

| Variable       | Required | Default       | Description |
|----------------|----------|---------------|-------------|
| `LLM_API_KEY`  | No       | (empty)       | API key. When empty, the Copilot runs in mock mode. Get a free key at https://console.groq.com |
| `LLM_MODEL`    | No       | `qwen/qwen3.8-27b` | Model identifier. |
| `LLM_BASE_URL` | No       | `https://api.groq.com/openai/v1` | Base URL. Change to use a different OpenAI-compatible provider. |

Mock mode works without an API key and is used in development and CI. It calls the prediction tool with default values to verify the pipeline works end to end.

## DVC Dataset Versioning

The dataset (`data/raw.csv`) is tracked with [DVC](https://dvc.org/) and stored on [DagsHub](https://dagshub.com/).

DVC is entirely optional: the demo runs without it, since `data/raw.csv` ships with the repo.

### Setup (local versioning, no credentials)

```bash
make dvc-init
```

This initializes DVC, registers the DagsHub remote, and tracks `data/raw.csv`. No token
is required. Local commands (`dvc status`, `dvc checkout`, `dvc diff`) work from here.

### Setup (with remote push/pull)

`DAGSHUB_TOKEN` is only needed to push to or pull from the DagsHub remote:

```bash
# 1. Get a DagsHub access token from https://dagshub.com/user/settings/tokens

# 2. Set the token as an environment variable
export DAGSHUB_TOKEN=<your-token>

# 3. Re-run the setup so the credentials land in .dvc/config.local (gitignored)
make dvc-init

# 4. Push the dataset
make dvc-push
```

### How it works

- `data/raw.csv` is the actual dataset file (gitignored)
- `data/raw.csv.dvc` is a small pointer file tracked by git (contains the file hash)
- DVC stores the full dataset on DagsHub's S3-compatible storage
- On another machine, run `make dvc-pull` to download the dataset (needs `DAGSHUB_TOKEN`)

## MLOps Workflow

1. **Train** -- `POST /api/retrain` logs metrics, parameters, and the model to MLflow
2. **Predict** -- `POST /api/predict` uses the latest model from the MLflow registry
3. **Feedback** -- `POST /api/feedback` adds the corrected label to the working dataset
4. **Retrain** -- `POST /api/retrain` trains a new model version on the updated dataset
5. **Drift** -- `POST /api/simulate-drift` injects drifted data and generates an Evidently report (no auto-retrain)
6. **Monitor** -- Open the Evidently dashboard or the generated report to inspect detected drift
7. **Reset** -- `POST /api/reset-data` restores the dataset to its original state
8. **Chat** -- `POST /api/chat` sends a natural-language query to the Copilot (optional LLMOps layer)

## LLM Evaluation

A small evaluation suite in `eval/` tests the Copilot against representative cases:

```bash
# Run against the live API (requires the stack to be running)
python eval/run_eval.py --base-url https://localhost
```

The script checks: correct tool invocation, grounded predictions, model-card limitation mentions, causal claim refusal, and historical-demo disclaimers.

## Development

```bash
# Install dev dependencies (ruff, pytest, httpx)
uv sync --extra dev

# Lint
uv run ruff check services/ tests/

# Format check
uv run ruff format --check services/ tests/

# Run tests
uv run pytest tests/ -v
```

## Environment Variables

See [`.env.example`](.env.example) for all required configuration.

## Tech Stack

- **API**: FastAPI + scikit-learn + Uvicorn
- **Tracking**: MLflow + PostgreSQL + MinIO
- **Monitoring**: Evidently AI
- **Copilot**: OpenAI-compatible LLM (optional)
- **Proxy**: Nginx (self-signed HTTPS)
- **Data versioning**: DVC + DagsHub
- **CI**: GitHub Actions + Ruff + pytest
- **Orchestration**: Docker Compose