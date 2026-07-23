.PHONY: up down build logs clean restart health retrain predict drift dataset-info model-info reset-data dvc-init dvc-push dvc-pull

# ======================================================
#  Titanic MLOps - Makefile
# ======================================================

# --- Main commands ---

up:
	@echo "Building and starting Titanic MLOps services..."
	docker compose up --build -d
	@echo ""
	@echo "All services are starting."
	@echo "   The model is NOT auto-trained. Use the frontend or 'make retrain' to train it."
	@echo ""
	@echo "Service URLs:"
	@echo "   Frontend:       https://localhost"
	@echo "   API Docs:       https://localhost/api/docs"
	@echo "   MLflow UI:      http://localhost:5001  (click Experiments in the left sidebar)"
	@echo "   Monitoring:     https://localhost/monitoring/"
	@echo "   MinIO Console:  http://localhost:9001"
	@echo ""
	@echo "API Endpoints:"
	@echo "   GET  /api/health          Health check"
	@echo "   POST /api/predict         Predict survival"
	@echo "   POST /api/feedback        Submit corrected label"
	@echo "   POST /api/retrain         Train or retrain the model"
	@echo "   POST /api/simulate-drift  Inject drifted data + generate drift report"
	@echo "   POST /api/reset-data      Restore original dataset"
	@echo "   GET  /api/model-info      Current model version"
	@echo "   GET  /api/dataset-info    Dataset statistics"
	@echo "   POST /api/chat            Chat with the Copilot"
	@echo "   GET  /api/traces          Recent LLM interaction traces"
	@echo ""
	@echo "Note: HTTPS uses self-signed certs. Accept the browser warning or use curl -k."

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build -d

build:
	docker compose build

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

clean:
	docker compose down -v --rmi local --remove-orphans
	@echo "Cleaned up all containers, volumes, and local images."

# --- API shortcuts ---

health:
	@curl -sk https://localhost/api/health | python3 -m json.tool

retrain:
	@echo "Triggering model training..."
	@curl -sk -X POST https://localhost/api/retrain | python3 -m json.tool

predict:
	@echo "Predicting survival for a 1st-class female, 29yo..."
	@curl -sk -X POST https://localhost/api/predict \
		-H "Content-Type: application/json" \
		-d '{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":100,"Embarked":"S"}' \
		| python3 -m json.tool

drift:
	@echo "Simulating data drift (500 samples)..."
	@curl -sk -X POST "https://localhost/api/simulate-drift?n_samples=500" | python3 -m json.tool

reset-data:
	@echo "Resetting dataset to original..."
	@curl -sk -X POST https://localhost/api/reset-data | python3 -m json.tool

dataset-info:
	@curl -sk https://localhost/api/dataset-info | python3 -m json.tool

model-info:
	@curl -sk https://localhost/api/model-info | python3 -m json.tool

# --- DVC ---

dvc-init:
	@bash scripts/setup_dvc.sh

dvc-push:
	dvc push -r origin

dvc-pull:
	dvc pull -r origin
