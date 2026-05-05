from flask import Flask, request, jsonify, render_template
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, set_access_cookies, unset_jwt_cookies, get_jwt_identity
import torch
from datetime import datetime
from functools import wraps
from models import db, User, Transaction
from config import Config
from model import load_fraud_model

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
jwt = JWTManager(app)

import json
import os
import numpy as np

def admin_required():
    def wrapper(fn):
        @wraps(fn)
        @jwt_required()
        def decorator(*args, **kwargs):
            current_user_id = int(get_jwt_identity())
            user = User.query.get(current_user_id)
            if not user or not user.is_admin:
                if request.path.startswith('/api/'):
                    return jsonify({"error": "Admin access required"}), 403
                return "Admin access required", 403
            return fn(*args, **kwargs)
        return decorator
    return wrapper


# Load global model and stats
PURPOSE_MAP = {
    'Payment': 0, 'Transfer': 1, 'Bill': 2, 'Gift': 3, 
    'Refund': 4, 'Rent': 5, 'Utilities': 6, 'Payroll': 7, 
    'Loan': 8, 'Other': 9
}

stats = {"mean": 0.0, "std": 1.0}
if os.path.exists("stats.json"):
    with open("stats.json", "r") as f:
        stats = json.load(f)

fraud_model = load_fraud_model(nhead=2)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=str(user.id))
    resp = jsonify({'login': True, 'user_id': user.id, 'is_admin': user.is_admin})
    set_access_cookies(resp, access_token)
    return resp, 200

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists"}), 400

    new_user = User(username=username)
    new_user.set_password(password)
    new_user.balance = 1000.0 # Starting balance for new accounts
    
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"msg": "User created successfully"}), 201

@app.route('/api/logout', methods=['POST'])
def logout():
    resp = jsonify({'logout': True})
    unset_jwt_cookies(resp)
    return resp, 200

@app.route('/api/me', methods=['GET'])
@jwt_required()
def me():
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    return jsonify({
        "id": user.id,
        "username": user.username,
        "balance": user.balance,
        "is_admin": user.is_admin
    })

@app.route('/api/transfer', methods=['POST'])
@jwt_required()
def transfer():
    current_user_id = int(get_jwt_identity())
    data = request.json
    
    receiver_username = data.get('receiver_username')
    amount = float(data.get('amount', 0))
    purpose = data.get('purpose', '')
    
    sender = User.query.get(current_user_id)
    receiver = User.query.filter_by(username=receiver_username).first()
    
    if not receiver:
        return jsonify({"error": "Receiver not found"}), 404
        
    if sender.balance < amount:
        return jsonify({"error": "Insufficient funds"}), 400

    now = datetime.utcnow()
    
    # 1. Fetch last 9 transactions for the sender to build the sequence
    last_txns = Transaction.query.filter_by(sender_id=sender.id).order_by(Transaction.timestamp.desc()).limit(9).all()
    last_txns.reverse() # chronologically: oldest to newest
    
    # 2. Build feature sequence
    # Features: [Z-Score Amount, hour, minute, day_of_week, time_delta]
    feat_seq, purp_seq = [], []
    prev_time = None
    
    # Combined list for processing
    items = last_txns + [{"amount": amount, "timestamp": now, "purpose": purpose}]
    
    for item in items:
        # Handle both SQLAlchemy objects and the current dict
        ts = item.timestamp if hasattr(item, 'timestamp') else item['timestamp']
        amt = item.amount if hasattr(item, 'amount') else item['amount']
        purp = item.purpose if hasattr(item, 'purpose') else item['purpose']
        
        hour = ts.hour
        minute = ts.minute
        day_of_week = ts.weekday()
        time_delta = (ts - prev_time).total_seconds() if prev_time else 0.0
        prev_time = ts
        
        # Capitalize purpose to match PURPOSE_MAP (e.g. 'transfer' -> 'Transfer')
        purp_capitalized = purp.title() if isinstance(purp, str) else purp
        
        z_amt = (amt - stats['mean']) / (stats['std'] + 1e-6)
        feat_seq.append([z_amt, hour, minute, day_of_week, time_delta])
        purp_seq.append(PURPOSE_MAP.get(purp_capitalized, 9))
        
    # Pad sequence if less than 10
    while len(feat_seq) < 10:
        feat_seq.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
        purp_seq.insert(0, 9)
        
    # 3. Run Inference
    device = next(fraud_model.parameters()).device
    f_tensor = torch.tensor([feat_seq], dtype=torch.float32).to(device)
    p_tensor = torch.tensor([purp_seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        fraud_prob = fraud_model(f_tensor, p_tensor).item()
        
    # Hybrid AI + Heuristic System:
    # The Transformer can over-index on temporal patterns (velocity) and miss single-point outliers.
    # We apply a rule-based boost for extreme Whale anomalies (Z-Score > 4.0)
    current_z_score = feat_seq[-1][0]
    if current_z_score > 4.0:
        fraud_prob = max(fraud_prob, 0.95)
        
    is_fraud = fraud_prob > 0.25
    
    # 4. Create Transaction
    new_txn = Transaction(
        sender_id=sender.id,
        receiver_id=receiver.id,
        amount=amount,
        timestamp=now,
        purpose=purpose,
        is_fraud=is_fraud,
        status='active' if not is_fraud else 'reversed'
    )
    
    if not is_fraud:
        sender.balance -= amount
        receiver.balance += amount
        msg = "Transfer successful"
    else:
        # If fraud, we block the funds transfer (or auto-reverse it)
        msg = "Transfer blocked due to high fraud probability"

    db.session.add(new_txn)
    db.session.commit()
    
    return jsonify({
        "msg": msg,
        "transaction_id": new_txn.id,
        "is_fraud": is_fraud,
        "fraud_score": fraud_prob
    }), 200

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET'])
@admin_required()
def admin_page():
    return render_template('admin.html')

@app.route('/api/admin/transactions', methods=['GET'])
@admin_required()
def admin_transactions():
    txns = Transaction.query.order_by(Transaction.timestamp.desc()).limit(50).all()
    result = []
    for t in txns:
        result.append({
            "id": t.id,
            "sender_id": t.sender_id,
            "receiver_id": t.receiver_id,
            "amount": t.amount,
            "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": t.purpose,
            "status": t.status,
            "is_fraud": t.is_fraud
        })
    return jsonify(result), 200
    
@app.route('/api/admin/users', methods=['GET'])
@admin_required()
def admin_users():
    users = User.query.all()
    result = []
    for u in users:
        result.append({
            "id": u.id,
            "username": u.username,
            "balance": u.balance,
            "is_admin": u.is_admin
        })
    return jsonify(result), 200

@app.route('/api/admin/reverse/<int:txn_id>', methods=['POST'])
@admin_required()
def reverse_transaction(txn_id):
    txn = Transaction.query.get(txn_id)
    if not txn:
        return jsonify({"error": "Transaction not found"}), 404
        
    if txn.status == 'reversed':
        return jsonify({"error": "Already reversed"}), 400

    # Mark as reversed
    txn.status = 'reversed'
    
    # Create compensating transaction (Receiver sends back to Sender)
    comp_txn = Transaction(
        sender_id=txn.receiver_id,
        receiver_id=txn.sender_id,
        amount=txn.amount,
        timestamp=datetime.utcnow(),
        purpose=f"REVERSAL of Txn #{txn.id}",
        status='active',
        is_fraud=False
    )
    
    # Adjust balances back
    sender = User.query.get(txn.sender_id)
    receiver = User.query.get(txn.receiver_id)
    
    sender.balance += txn.amount
    receiver.balance -= txn.amount
    
    db.session.add(comp_txn)
    db.session.commit()
    
    return jsonify({"msg": "Transaction reversed", "compensating_txn_id": comp_txn.id}), 200

@app.route('/api/admin/explain/<int:txn_id>', methods=['GET'])
@admin_required()
def explain_transaction(txn_id):
    txn = Transaction.query.get(txn_id)
    if not txn:
        return jsonify({"error": "Not found"}), 404
        
    # Reconstruct the sequence used for this transaction
    last_txns = Transaction.query.filter(
        Transaction.sender_id == txn.sender_id,
        Transaction.timestamp < txn.timestamp
    ).order_by(Transaction.timestamp.desc()).limit(9).all()
    last_txns.reverse()
    
    feat_seq, purp_seq = [], []
    prev_time = None
    items = last_txns + [txn]
    for item in items:
        dt = (item.timestamp - prev_time).total_seconds() if prev_time else 0.0
        dt = float(np.log1p(dt))  # Log-Temporal Encoding: compress scale for velocity burst detection
        prev_time = item.timestamp
        z_amt = (item.amount - stats['mean']) / (stats['std'] + 1e-6)
        feat_seq.append([z_amt, item.timestamp.hour, item.timestamp.minute, item.timestamp.weekday(), dt])
        purp_seq.append(PURPOSE_MAP.get(item.purpose, 9))
        
    while len(feat_seq) < 10:
        feat_seq.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
        purp_seq.insert(0, 9)

    device = next(fraud_model.parameters()).device
    f_tensor = torch.tensor([feat_seq], dtype=torch.float32).to(device)
    p_tensor = torch.tensor([purp_seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        _, attn_weights = fraud_model(f_tensor, p_tensor, return_attention=True)
    
    # Determine reasoning message
    reason = "Transaction appears normal."
    current_z_score = feat_seq[-1][0]
    
    if txn.is_fraud:
        if current_z_score > 4.0:
            reason = "Whale Attack Detected: The transaction amount is an extreme outlier compared to the user's history."
        else:
            reason = "Velocity Attack Detected: The model identified a rapid burst of transactions in a short time window."

    # Return attention weights for the 10 steps
    return jsonify({
        "transaction_id": txn_id,
        "attention": attn_weights.tolist(),
        "history": [f"Txn #{t.id}: ${t.amount}" for t in last_txns] + [f"Current: ${txn.amount}"],
        "reason": reason
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Listen on 0.0.0.0 to allow access from AWS EC2 Public IP
    app.run(debug=True, host='0.0.0.0', port=5000)
