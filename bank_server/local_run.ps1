Write-Host "Setting up Local Development Environment..."

# Create virtual environment if it doesn't exist
if (-Not (Test-Path "venv")) {
    python -m venv venv
}

# Activate virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run bootstrap (generates DB, trains model)
if (-Not (Test-Path "model.pth")) {
    Write-Host "Running bootstrap process (this might take a few minutes for 50k rows)..."
    python bootstrap.py
}

# Start the Flask development server
Write-Host "Starting Flask Server on http://127.0.0.1:5000..."
python app.py
