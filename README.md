# Financial Fraud Detection (CC Lab)

Implementation workspace for the CC Lab project based on the paper:
**“A Distributed Knowledge Distillation Framework for Financial Fraud Detection Based on Transformer”** (IEEE Access, 2024).

This repository contains a full-stack, research-grade banking simulator and fraud detection engine. It uses a Teacher-Student Knowledge Distillation architecture to train a fast, lightweight Transformer model capable of detecting complex financial fraud patterns (such as "Whale" and "Velocity" attacks) in real-time.

## Features

- **Transformer-Based Inference:** Utilizes a custom PyTorch Sequence Transformer with Sinusoidal Positional Encoding to catch rhythmic bot attacks and massive outliers.
- **Knowledge Distillation:** A 6-head Teacher model transfers its "dark knowledge" to a high-speed 2-head Student model optimized for <1ms inference latency.
- **Hybrid AI Heuristics:** Combines deep learning with robust rule-based logic (Z-Score > 4.0 detection) to prevent sequence bias and catch single-point anomalies.
- **Explainable AI (XAI):** Features an Admin Dashboard that provides IEEE-grade Attention Maps, showing exactly which transactions triggered the fraud alert and why.
- **Role-Based Access Control:** Secure JWT authentication protecting the administrative endpoints and XAI tools.

## Architecture

1. **Backend Server:** Flask-based REST API (`bank_server/app.py`)
2. **Database:** SQLAlchemy (SQLite by default, scalable to PostgreSQL)
3. **Machine Learning Model:** PyTorch Sequence Transformer (`bank_server/model.py`)
4. **Training Loop:** Automated data generation and distillation (`bank_server/bootstrap.py`)
5. **Evaluation:** Sklearn-based metrics generation (Precision, Recall, F1, Confusion Matrix) (`bank_server/evaluate.py`)

## Getting Started

### 1. Setup the Environment

Ensure you have Python 3.9+ installed.

```bash
cd bank_server
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Bootstrap & Train the Model

The bootstrap script will generate 50,000 synthetic transactions (including Whale and Velocity attacks), compute statistical baselines (Z-Scores), and execute the Teacher-Student distillation training loop.

```bash
python bootstrap.py
```

### 3. Run the Banking Server

Start the Flask application:

```bash
.\local_run.ps1
```
The server will be available at `http://localhost:5000`.

### 4. Test the System

You can test the system in two ways:

**Manual Web UI:**
1. Navigate to `http://localhost:5000`
2. Login as a new user or the admin (Credentials: `admin` / `adminpass`)
3. Execute transfers and view the XAI dashboard in the Admin Panel.

**Automated Simulator:**
Use the provided script to launch automated attacks against the server:
```bash
python simulator.py --strategy velocity --target_user user_2 --burst_count 12
python simulator.py --strategy whale --target_user user_3
```

## Evaluation Artifacts

To generate the confusion matrix and precision-recall curves for your research paper or presentation, run:

```bash
python evaluate.py
```
The resulting PNGs will be saved in the `bank_server/static/` directory.
