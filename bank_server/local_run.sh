#!/bin/bash
echo "Setting up Local Development Environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run bootstrap (generates DB, trains model)
if [ ! -f "model.pth" ]; then
    echo "Running bootstrap process (this might take a few minutes for 50k rows)..."
    python bootstrap.py
fi

# Start the Flask development server
echo "Starting Flask Server on http://127.0.0.1:5000..."
python app.py
