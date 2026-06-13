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

# --- Set credentials (stored in .dvc/config.local - gitignored) ---
dvc remote modify origin --local auth basic
dvc remote modify origin --local user Lalanne0
dvc remote modify origin --local password 239aa8b3d33cfe48eb027e0db1b42e1ac612df34
echo "Credentials stored in .dvc/config.local"

# --- Track the dataset ---
if [ ! -f "raw.csv.dvc" ]; then
    dvc add raw.csv
    echo "raw.csv tracked by DVC"
else
    echo "raw.csv already tracked by DVC"
fi

echo ""
echo "DVC setup complete!"
echo ""
echo "   Next steps:"
echo "   1. dvc push -r origin        Push data to DagsHub"
echo "   2. git add raw.csv.dvc .dvc  Commit DVC tracking files"
echo "   3. git commit -m 'Add DVC tracking'"
echo ""
echo "   To pull data on another machine:"
echo "   1. dvc pull -r origin"
