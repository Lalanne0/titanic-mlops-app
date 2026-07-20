FROM python:3.12-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/api/app/ ./app/

# Bundle the dataset (copied to volume on first startup)
COPY data/raw.csv /app/raw_original.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
