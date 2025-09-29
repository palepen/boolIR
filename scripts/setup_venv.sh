#!/bin/bash
# scripts/setup_venv.sh

set -e

echo ">>> Creating virtual environment..."
python3 -m venv .venv
echo "✅ Created .venv/"

echo ">>> Activating and upgrading pip..."
source .venv/bin/activate
python -m pip install --upgrade pip

echo ">>> Installing Python dependencies..."
pip install -r scripts/requirements.txt

echo ""
echo "🎉 Done! Activate your environment with:"
echo "    source .venv/bin/activate"
echo ""
echo "💡 Verify GPU support:"
echo "    python -c \"import onnxruntime as ort; print('Providers:', ort.get_available_providers())\""