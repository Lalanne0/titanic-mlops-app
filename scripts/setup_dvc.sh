#!/bin/bash
# ======================================================
#  DVC setup for DagsHub remote storage
# ======================================================
set -e

cd "$(dirname "$0")/.."

echo "Setting up DVC with DagsHub remote..."
echo ""

# --- Initialize DVC ---
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC initialized"
else
    echo "DVC already initialized"
fi

# --- Add DagsHub remote ---
dvc remote add -f origin https://dagshub.com/Lalanne0/titanic-mlops-app.dvc
echo "DagsHub remote 'origin' configured"

# --- Set credentials (optional, stored in .dvc/config.local - gitignored) ---
# The token is only needed to push to or pull from the DagsHub remote. Local
# versioning (dvc add, dvc status, dvc checkout) works without any credentials.
DAGSHUB_USER="${DAGSHUB_USER:-Lalanne0}"
HAS_TOKEN=0
if [ -n "${DAGSHUB_TOKEN}" ]; then
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "${DAGSHUB_USER}"
    dvc remote modify origin --local password "${DAGSHUB_TOKEN}"
    HAS_TOKEN=1
    echo "Credentials stored in .dvc/config.local"
else
    echo "DAGSHUB_TOKEN is not set - skipping remote credentials (local DVC only)."
    echo "   To enable push/pull later, get a token from"
    echo "   https://dagshub.com/user/settings/tokens then run:"
    echo "   export DAGSHUB_TOKEN=<your-token> && make dvc-init"
fi

# --- Track the dataset ---
if [ ! -f "data/raw.csv.dvc" ]; then
    dvc add data/raw.csv
    echo "data/raw.csv tracked by DVC"
else
    echo "data/raw.csv already tracked by DVC"
fi

echo ""
echo "DVC setup complete."
echo ""
echo "   Next steps:"
echo "   1. git add data/raw.csv.dvc .dvc  Commit DVC tracking files"
echo "   2. git commit -m 'Add DVC tracking'"
if [ "${HAS_TOKEN}" -eq 1 ]; then
    echo "   3. dvc push -r origin        Push data to DagsHub"
    echo ""
    echo "   To pull data on another machine:"
    echo "   1. dvc pull -r origin"
else
    echo ""
    echo "   dvc push / dvc pull need DAGSHUB_TOKEN. Without it the dataset stays"
    echo "   versioned locally only, which is enough to run the whole demo."
fi
