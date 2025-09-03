import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import NearMiss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# PyTorch Geometric libraries
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU, Dropout


# --- 1. DATA PREPARATION (with NearMiss and k-NN Graph) ---
print("Đang tải và tiền xử lý dữ liệu...")
df = pd.read_csv('D://Python//Do_An_Nganh//DataSet//Diabetes-Health-Indicators-main//Diabetes-Health-Indicators-main//diabetes_binary_health_indicators_BRFSS2015.csv')

# Tách đặc trưng (X) và nhãn (y)
X_df = df.drop('Diabetes_binary', axis=1)
y_series = df['Diabetes_binary']
best_features = ['GenHlth', 'PhysHlth', 'Income', 'DiffWalk', 'MentHlth', 'BMI', 'HighBP']
X_df = X_df[best_features]

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
    
# Áp dụng NearMiss để cân bằng dữ liệu (under-sampling)
print(f"Kích thước dữ liệu gốc: {X_scaled.shape}")
nm = NearMiss(version=1, n_neighbors=3)
X_resampled, y_resampled = nm.fit_resample(X_scaled, y_series)
print(f"Kích thước dữ liệu sau NearMiss: {X_resampled.shape}")

# Tạo đồ thị k-NN từ dữ liệu đã được under-sampled
k = 15
knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
knn_model.fit(X_resampled)
distances, indices = knn_model.kneighbors(X_resampled)
edge_list = []
for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        if i != neighbor:
            edge_list.append((i, neighbor))
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Chuyển đổi đặc trưng và nhãn sang tensor của PyTorch
x = torch.tensor(X_resampled, dtype=torch.float32)
y = torch.tensor(y_resampled.values, dtype=torch.float32)

# Tạo đối tượng PyTorch Geometric Data
data = Data(x=x, edge_index=edge_index, y=y)

# Tạo các mask để chia dữ liệu thành 3 tập
num_nodes = data.num_nodes
train_mask, temp_mask = train_test_split(np.arange(num_nodes), test_size=0.3, stratify=y, random_state=42)
val_mask, test_mask = train_test_split(temp_mask, test_size=0.5, stratify=y.numpy()[temp_mask], random_state=42)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_mask] = True
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask[val_mask] = True
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask[test_mask] = True

print("PyTorch Geometric Data object created successfully with under-sampled data:")
print(data)

# --- 2. GIN MODEL ARCHITECTURE ---
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        # Sử dụng MLP với GINConv
        self.conv1 = GINConv(Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        ))
        
        self.output_head = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(p=0.5),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        return self.output_head(x)

# --- 3. TRAINING AND EVALUATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(in_channels=data.num_node_features, hidden_channels=64, out_channels=1).to(device)
data = data.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze(-1)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_all_metrics(mask):
    model.eval()
    out = model(data.x, data.edge_index).squeeze(-1)
    
    # Chuyển đổi đầu ra thành dự đoán nhị phân
    y_true = data.y[mask].cpu().numpy()
    y_pred_probs = torch.sigmoid(out[mask]).cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(float)

    # Tính toán các chỉ số
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)
    
    return acc, prec, rec, f1, auc

num_epochs = 200
patience = 15
best_val_accuracy = 0.0
epochs_without_improvement = 0
best_model_state = None

print("\nStarting GIN model training with Early Stopping...")
for epoch in range(num_epochs):
    loss = train()
    val_acc, _, _, _, _ = evaluate_all_metrics(data.val_mask)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Val Acc: {val_acc:.4f} -> New best!")
    else:
        epochs_without_improvement += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
        if epochs_without_improvement >= patience:
            print(f"  -> Validation accuracy has not improved for {patience} epochs. Stopping early!")
            break

print("\nGIN training complete!")

if best_model_state:
    print(f"\nLoading best model with validation accuracy: {best_val_accuracy:.4f}")
    model.load_state_dict(best_model_state)
    
    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_all_metrics(data.test_mask)
    
    print(f"--- Kết quả trên tập kiểm tra với mô hình tốt nhất ---")
    print(f"Accuracy (Độ chính xác): {test_acc:.4f}")
    print(f"Precision (Độ chính xác dự đoán dương tính): {test_prec:.4f}")
    print(f"Recall (Độ nhạy): {test_rec:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"Điểm AUC (AUC): {test_auc:.4f}")
