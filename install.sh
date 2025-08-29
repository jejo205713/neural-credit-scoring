#!/bin/bash

echo "[INFO] Starting Neural Credit Collective Prototype setup..."

# Step 1: Check Python version
PYTHON_VERSION=$(python3 -V 2>&1)
echo "[INFO] Using Python: $PYTHON_VERSION"

# Step 2: Create virtual environment
echo "[INFO] Creating virtual environment 'venv'..."
python3 -m venv venv

# Step 3: Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Step 5: Install requirements
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Step 6: Verify installation
echo "[INFO] Verifying installed packages..."
pip list

echo "[SUCCESS] Installation complete!"
echo
echo "============================"
echo "To run the Neural Credit Collective Prototype:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the prototype:"
echo "   python run_prototype.py"
echo "============================"

