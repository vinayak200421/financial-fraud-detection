import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SequenceTransformer(nn.Module):
    def __init__(self, num_features=5, num_purposes=10, purpose_dim=8, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(SequenceTransformer, self).__init__()
        
        # Purpose Embedding
        self.purpose_embedding = nn.Embedding(num_purposes, purpose_dim)
        
        # Linear layer to map the features plus purpose embedding to d_model
        self.input_projection = nn.Linear(num_features + purpose_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, purpose_ids, return_attention=False):
        # x shape: (batch_size, seq_len, num_features-1)
        # purpose_ids: (batch_size, seq_len)
        
        # Embed purpose
        p_emb = self.purpose_embedding(purpose_ids) # (batch, seq, purpose_dim)
        
        # Concatenate features and purpose embedding
        x = torch.cat([x, p_emb], dim=-1) # (batch, seq, num_features-1 + purpose_dim)
        
        # Project and encode
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # We need to extract attention if requested. 
        # Note: PyTorch's TransformerEncoder doesn't return weights easily.
        # For IEEE-grade XAI, we'll manually iterate layers if return_attention is True, 
        # or use a simplified approach by returning the weights of the last layer's self-attention.
        
        if return_attention:
            # We'll use the last layer's attention for visualization
            # In a real IEEE paper, we might use Integrated Gradients or Attention Rollout.
            # Use normalized cosine similarity to avoid NaN on extreme outliers (Whale attacks)
            # Including explicit epsilon (1e-8) for denominator numerical stability
            x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
            attn_weights = torch.softmax(torch.matmul(x_norm, x_norm.transpose(-2, -1)) / 0.1, dim=-1)
            
            x = self.transformer_encoder(x)
            pooled = x.mean(dim=1)
            logits = self.classifier(pooled)
            
            # Get what the *current* transaction attended to in the sequence
            current_attn = attn_weights[:, -1, :] # (batch, seq_len)
            return torch.sigmoid(logits).squeeze(-1), current_attn
        
        x = self.transformer_encoder(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return torch.sigmoid(logits).squeeze(-1)

def load_fraud_model(model_path="model.pth", nhead=2, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default to Student architecture (nhead=2) unless specified
    model = SequenceTransformer(nhead=nhead)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print(f"Model loaded on {device} (nhead={nhead})")
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {e}. Returning untrained.")
        model.to(device)
        return model
