# Link tải lại dataset nếu cần:
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("D://Python//Do_An_Nganh//DataSet//Diabetes-Health-Indicators-main//Diabetes-Health-Indicators-main//diabetes_binary_health_indicators_BRFSS2015.csv")
X = df.drop(columns=["Diabetes_binary"]).values
y = df["Diabetes_binary"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512)

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)

# Transformer model
class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_blocks(x)
        return self.head(x)

model = TabTransformer(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
def train_model(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            output = model(xb)
            y_true.extend(yb.numpy())
            y_pred.extend((output.numpy() > 0.5).astype(int))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)

# Run training
for epoch in range(10):
    train_model(model, train_dl, optimizer, criterion)
    acc, f1 = evaluate(model, test_dl)
    print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
