import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from app import app, db
from models import Transaction
from model import load_fraud_model

PURPOSE_MAP = {
    'Payment': 0, 'Transfer': 1, 'Bill': 2, 'Gift': 3, 
    'Refund': 4, 'Rent': 5, 'Utilities': 6, 'Payroll': 7, 
    'Loan': 8, 'Other': 9
}

def build_test_data(stats):
    print("Fetching transactions for IEEE Evaluation...")
    all_txns = Transaction.query.order_by(Transaction.timestamp.desc()).limit(5000).all()
    all_txns.reverse()

    user_histories = {}
    X_feat, X_purp, y = [], [], []
    
    # Pre-populate histories
    for txn in Transaction.query.order_by(Transaction.timestamp).limit(45000).all():
        if txn.sender_id not in user_histories:
            user_histories[txn.sender_id] = []
        user_histories[txn.sender_id].append(txn)

    for txn in all_txns:
        sender_id = txn.sender_id
        history = user_histories.get(sender_id, [])[-9:]
        
        feat_seq, purp_seq = [], []
        prev_time = None
        items = history + [txn]
        for item in items:
            dt = (item.timestamp - prev_time).total_seconds() if prev_time else 0.0
            prev_time = item.timestamp
            z_amt = (item.amount - stats['mean']) / (stats['std'] + 1e-6)
            feat_seq.append([z_amt, item.timestamp.hour, item.timestamp.minute, item.timestamp.weekday(), dt])
            purp_seq.append(PURPOSE_MAP.get(item.purpose, 9))
            
        while len(feat_seq) < 10:
            feat_seq.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
            purp_seq.insert(0, 9)
            
        X_feat.append(feat_seq)
        X_purp.append(purp_seq)
        y.append(1.0 if txn.is_fraud else 0.0)
        
    return (torch.tensor(X_feat, dtype=torch.float32), 
            torch.tensor(X_purp, dtype=torch.long), 
            torch.tensor(y, dtype=torch.float32))

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fraud_model(model_path="model.pth", nhead=2, device=device)
    
    if not os.path.exists("stats.json"):
        print("Error: stats.json not found. Run bootstrap first.")
        return
        
    with open("stats.json", "r") as f:
        stats = json.load(f)

    X_f, X_p, y_true = build_test_data(stats)
    X_f, X_p = X_f.to(device), X_p.to(device)
    
    print("Running Inference for Metrics...")
    with torch.no_grad():
        y_probs = model(X_f, X_p).cpu().numpy()
        y_pred = (y_probs > 0.90).astype(float)
        y_true = y_true.numpy()

    # Create static directory
    os.makedirs("static", exist_ok=True)

    print("\n" + "="*40)
    print("RESEARCH-GRADE EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix - Sequence Transformer')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('static/confusion_matrix.png')
    print("Saved: static/confusion_matrix.png")

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Student Transformer')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('static/precision_recall_curve.png')
    print("Saved: static/precision_recall_curve.png")
    print("="*40)

if __name__ == "__main__":
    with app.app_context():
        evaluate()
