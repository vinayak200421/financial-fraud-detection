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
            dt = float(np.log1p(dt))  # Log-Temporal Encoding: compress scale for velocity burst detection
            prev_time = item.timestamp
            z_amt = (item.amount - stats['mean']) / (stats['std'] + 1e-6)
            z_amt = max(min(z_amt, 10.0), -10.0)
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
        
        # Apply Hybrid Heuristic
        z_scores = X_f[:, -1, 0].cpu().numpy()
        for i in range(len(y_probs)):
            if z_scores[i] > 4.0:
                y_probs[i] = max(y_probs[i], 0.95)
                
        y_pred = (y_probs > 0.25).astype(float)
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

    precision_val = precision_score(y_true, y_pred)
    recall_val    = recall_score(y_true, y_pred)
    f1_val        = f1_score(y_true, y_pred)
    acc_val       = accuracy_score(y_true, y_pred)

    # 2. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    auc_pr = np.trapezoid(precision_curve[::-1], recall_curve[::-1]) if hasattr(np, 'trapezoid') else np.trapz(precision_curve[::-1], recall_curve[::-1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='#2196F3', linewidth=2, label=f'Student Transformer (AUC={auc_pr:.3f})')
    plt.fill_between(recall_curve, precision_curve, alpha=0.1, color='#2196F3')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curve — Hybrid Transformer (IEEE)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/precision_recall_curve.png', dpi=150)
    print("Saved: static/precision_recall_curve.png")

    # 3. Metrics Bar Chart (Accuracy, Precision, Recall, F1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values  = [acc_val, precision_val, recall_val, f1_val]
    colors  = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    plt.figure(figsize=(9, 6))
    bars = plt.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.ylabel('Score', fontsize=13)
    plt.title('Evaluation Metrics — Student Transformer (IEEE)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/metrics_bar_chart.png', dpi=150)
    print("Saved: static/metrics_bar_chart.png")

    # 4. ROC Curve
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_roc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#E91E63', linewidth=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#E91E63')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve — Hybrid Transformer (IEEE)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curve.png', dpi=150)
    print("Saved: static/roc_curve.png")

    print("="*40)

if __name__ == "__main__":
    with app.app_context():
        evaluate()
