#!/bin/bash

# Define the virtual environment directory
VENV_DIR="thesis_venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment exists. Updating dependencies..."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install/upgrade requirements
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing/upgrading dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Done!"
