# Flask Bank Server with Stateful Transformer Fraud Engine

This project simulates a live banking environment where transactions are continuously evaluated for fraud using a sequence-based PyTorch Transformer model.

## Features
- **Core Backend:** Flask API with SQLAlchemy (`User` and `Transaction` models).
- **Fraud Engine:** Stateful Transformer Encoder evaluating a sequence of the last 10 transactions.
- **SSO:** JWT-based login stored in a 3-day HttpOnly cookie.
- **Simulators:** Built-in Whale (large transaction) and Velocity (rapid micro-transactions) fraud simulators.
- **Admin Panel:** UI to monitor transactions and execute Reversals.

## Setup Instructions (Local)
1. Run `bash local_run.sh`. This will:
   - Create a virtual environment and install dependencies from `requirements.txt`.
   - Run `bootstrap.py` which provisions an SQLite DB, generates 50,000 synthetic transactions (with SMOTE/logic for 5-10% fraud), and trains the Transformer model (`model.pth`) to reach paper-level accuracy.
   - Start the Flask dev server on `http://127.0.0.1:5000`.
2. Access the Admin Panel at `http://127.0.0.1:5000/admin`.
3. To test the simulators, run: `python simulator.py --strategy both`

## Setup Instructions (Azure Linux VM)
1. On your local machine, run `bash deploy_azure_vm.sh` to provision an Ubuntu VM using the Azure CLI.
2. SCP this `bank_server` directory to the VM.
3. SSH into the VM and run `sudo bash vm_setup.sh`.
4. The script will automatically install Nginx, PostgreSQL, setup Gunicorn as a systemd service, and configure the reverse proxy. 
5. Access your Bank Server via the VM's Public IP Address.
