FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY services/mlflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Shell form so ${…} env vars are expanded at runtime
CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root s3://mlflow/ \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts '*'
