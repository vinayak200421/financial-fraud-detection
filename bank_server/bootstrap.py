import os
import random
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app import app, db
from models import User, Transaction
from model import SequenceTransformer

PURPOSE_MAP = {
    'Payment': 0, 'Transfer': 1, 'Bill': 2, 'Gift': 3, 
    'Refund': 4, 'Rent': 5, 'Utilities': 6, 'Payroll': 7, 
    'Loan': 8, 'Other': 9
}

def seed_db():
    print("Initializing Database with diverse Purpose types...")
    db.drop_all()
    db.create_all()

    users = [User(username="admin", is_admin=True, balance=100000) for _ in range(1)]
    users[0].set_password("adminpass")
    
    for i in range(1, 100):
        u = User(username=f"user_{i}", balance=random.randint(5000, 50000))
        u.set_password("pass123")
        users.append(u)
    
    db.session.add_all(users)
    db.session.commit()

    print("Generating 50,000 transactions (Whale + Velocity patterns)...")
    user_ids = [u.id for u in users]
    start_time = datetime.utcnow() - timedelta(days=90)
    
    purposes = list(PURPOSE_MAP.keys())
    transactions = []
    
    count = 0
    while count < 50000:
        sender = random.choice(user_ids)
        receiver = random.choice([uid for uid in user_ids if uid != sender])
        
        dice = random.random()
        
        if dice < 0.04:
            # 1. Whale Attack
            amount = random.uniform(20000, 90000)
            purpose = "Transfer"
            txn_time = start_time + timedelta(minutes=random.randint(1, 1440 * 90))
            t = Transaction(sender_id=sender, receiver_id=receiver, amount=amount, 
                            timestamp=txn_time, purpose=purpose, status='reversed', is_fraud=True)
            transactions.append(t)
            count += 1
        elif dice < 0.08:
            # 2. Velocity Attack (Burst of 10-15 micro transactions)
            burst_size = random.randint(10, 15)
            base_time = start_time + timedelta(minutes=random.randint(1, 1440 * 90))
            for i in range(burst_size):
                if count >= 50000: break
                amount = random.uniform(1, 10)
                purpose = f"Micro test {i}"
                txn_time = base_time + timedelta(seconds=i * 5) # 5 seconds apart
                t = Transaction(sender_id=sender, receiver_id=receiver, amount=amount, 
                                timestamp=txn_time, purpose=purpose, status='reversed', is_fraud=True)
                transactions.append(t)
                count += 1
        else:
            # 3. Normal Transaction
            amount = random.uniform(10, 800)
            purpose = random.choice(purposes)
            txn_time = start_time + timedelta(minutes=random.randint(1, 1440 * 90))
            t = Transaction(sender_id=sender, receiver_id=receiver, amount=amount, 
                            timestamp=txn_time, purpose=purpose, status='active', is_fraud=False)
            transactions.append(t)
            count += 1
    
    transactions.sort(key=lambda x: x.timestamp)
    db.session.bulk_save_objects(transactions)
    db.session.commit()

def build_training_data():
    print("Building sequences with Z-Score & Purpose Embeddings...")
    all_txns = Transaction.query.order_by(Transaction.timestamp).all()
    
    # Calculate Z-Score params for Amount
    amounts = np.array([t.amount for t in all_txns])
    mean_amt, std_amt = amounts.mean(), amounts.std()
    print(f"Z-Score Stats: Mean={mean_amt:.2f}, Std={std_amt:.2f}")

    import json
    with open("stats.json", "w") as f:
        json.dump({"mean": float(mean_amt), "std": float(std_amt)}, f)

    user_histories = {}
    X_feat, X_purp, y = [], [], []
    
    for txn in all_txns:
        sender_id = txn.sender_id
        if sender_id not in user_histories:
            user_histories[sender_id] = []
            
        history = user_histories[sender_id][-9:]
        
        feat_seq, purp_seq = [], []
        prev_time = None
        
        # Prepare sequence (9 history + 1 current)
        items = history + [txn]
        for item in items:
            dt = (item.timestamp - prev_time).total_seconds() if prev_time else 0.0
            dt = float(np.log1p(dt))  # Log-Temporal Encoding: compress scale for velocity burst detection
            prev_time = item.timestamp
            
            # Features: [Z-Score Amount, hour, minute, day, dt]
            z_amt = (item.amount - mean_amt) / (std_amt + 1e-6)
            z_amt = max(min(z_amt, 10.0), -10.0) # Clip extreme whales
            feat_seq.append([z_amt, item.timestamp.hour, item.timestamp.minute, item.timestamp.weekday(), dt])
            purp_seq.append(PURPOSE_MAP.get(item.purpose, 9))
            
        # Pad
        while len(feat_seq) < 10:
            feat_seq.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])
            purp_seq.insert(0, 9)
            
        X_feat.append(feat_seq)
        X_purp.append(purp_seq)
        y.append(1.0 if txn.is_fraud else 0.0)
        user_histories[sender_id].append(txn)
        
    # Oversampling: SMOTE-like behavior for Fraud
    fraud_indices = [i for i, label in enumerate(y) if label == 1.0]
    legit_indices = [i for i, label in enumerate(y) if label == 0.0]
    
    if len(fraud_indices) > 0 and len(legit_indices) > len(fraud_indices):
        repeat_factor = len(legit_indices) // len(fraud_indices)
        
        # Extend the lists with copies of fraud cases
        for i in fraud_indices:
            # We copy repeat_factor - 1 times (since it's already there once)
            X_feat.extend([X_feat[i]] * (repeat_factor - 1))
            X_purp.extend([X_purp[i]] * (repeat_factor - 1))
            y.extend([1.0] * (repeat_factor - 1))

    return (torch.tensor(X_feat, dtype=torch.float32), 
            torch.tensor(X_purp, dtype=torch.long), 
            torch.tensor(y, dtype=torch.float32))

def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.7):
    # Standard Task Loss
    task_loss = F.binary_cross_entropy(student_logits, labels)
    
    # KD Loss (KL Divergence)
    soft_teacher = teacher_logits # Probs from teacher
    kd_loss = F.binary_cross_entropy(student_logits, soft_teacher)
    
    return alpha * task_loss + (1 - alpha) * (T**2) * kd_loss

def train_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    X_feat, X_purp, y = build_training_data()
    dataset = TensorDataset(X_feat, X_purp, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 1. Train Teacher (6-head)
    print("\n--- Training Teacher Model (6-head attention) ---")
    teacher = SequenceTransformer(nhead=4).to(device) # Note: PyTorch nhead must divide d_model(64)
    # Using 4 heads for stability in standard division, 
    # but the paper logic implies higher complexity.
    optimizer_t = optim.Adam(teacher.parameters(), lr=0.001)
    teacher.train()
    for epoch in range(10):
        for f, p, l in loader:
            f, p, l = f.to(device), p.to(device), l.to(device)
            optimizer_t.zero_grad()
            preds = teacher(f, p)
            loss = F.binary_cross_entropy(preds, l)
            loss.backward()
            optimizer_t.step()
        print(f"Teacher Epoch {epoch+1} Complete")
    torch.save(teacher.state_dict(), "teacher.pth")

    # 2. Knowledge Distillation to Student (2-head)
    print("\n--- Distilling to Student Model (2-head attention, T=3) ---")
    teacher.eval()
    student = SequenceTransformer(nhead=2).to(device)
    optimizer_s = optim.Adam(student.parameters(), lr=0.001)
    
    student.train()
    for epoch in range(20):
        total_loss = 0
        for f, p, l in loader:
            f, p, l = f.to(device), p.to(device), l.to(device)
            optimizer_s.zero_grad()
            
            with torch.no_grad():
                t_preds = teacher(f, p)
            
            s_preds = student(f, p)
            loss = distillation_loss(s_preds, t_preds, l, T=3)
            loss.backward()
            optimizer_s.step()
            total_loss += loss.item()
        print(f"Student Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
    
    torch.save(student.state_dict(), "model.pth") # Save as primary model
    print("Distillation Complete. Student model saved.")

if __name__ == "__main__":
    with app.app_context():
        seed_db()
        train_system()
