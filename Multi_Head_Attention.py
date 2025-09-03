import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import NearMiss
import numpy as np

# ===================== CÁC HÀM TIỀN XỬ LÝ DỮ LIỆU =====================

def smart_feature_engineering(df):
    """
    Kỹ thuật feature engineering thông minh để tạo các đặc trưng mới
    dựa trên mối quan hệ giữa các đặc trưng gốc và kiến thức chuyên ngành.
    """
    df = df.copy()
    
    # Check for missing values in original data
    if df.isnull().sum().sum() > 0:
        print("Warning: Phát hiện giá trị thiếu, đang điền bằng giá trị trung vị.")
        df = df.fillna(df.median())
    
    # Top interaction features dựa trên domain knowledge
    df['BMI_HighBP'] = df['BMI'] * df['HighBP']
    df['BMI_HighChol'] = df['BMI'] * df['HighChol'] 
    df['Age_BMI'] = df['Age'] * df['BMI']
    
    # Health score with safe division
    health_sum = df['GenHlth'] + df['MentHlth'] + df['PhysHlth']
    df['Health_Score'] = health_sum / 3.0
    
    # BMI categories (medical knowledge) - handle edge cases
    try:
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100], 
                                   labels=[0, 1, 2, 3], 
                                   include_lowest=True).astype(float)
    except Exception as e:
        print(f"Lỗi khi phân loại BMI: {e}")
        df['BMI_Category'] = 0  # Gán giá trị mặc định nếu có lỗi
    
    # Handle any NaN created during feature engineering
    df = df.fillna(0)
    
    return df

# ===================== CHUẨN BỊ DỮ LIỆU ĐỂ HUẤN LUYỆN =====================

# Tải dataset
print("Đang tải và tiền xử lý dữ liệu...")
df = pd.read_csv('D://Python//Do_An_Nganh//DataSet//Diabetes-Health-Indicators-main//Diabetes-Health-Indicators-main//diabetes_binary_health_indicators_BRFSS2015.csv')

# Áp dụng feature engineering
df_enhanced = smart_feature_engineering(df)

# Lỗi đã được khắc phục ở đây
df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan).fillna(df_enhanced.median())

# Chọn lọc đặc trưng dựa trên kết quả từ notebook (hoặc bạn có thể thử 'all' để so sánh)
best_features = ['GenHlth', 'PhysHlth', 'Income', 'DiffWalk', 'MentHlth', 'BMI', 'HighBP']
X_df = df_enhanced[best_features]
y_series = df_enhanced['Diabetes_binary']

X = X_df.values
y = y_series.values

# Áp dụng NearMiss để cân bằng dữ liệu (under-sampling)
print(f"Kích thước dữ liệu gốc: {X.shape}")
nm = NearMiss(version=1, n_neighbors=10)
X_resampled, y_resampled = nm.fit_resample(X, y)
print(f"Kích thước dữ liệu sau NearMiss: {X_resampled.shape}")

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Chia dữ liệu thành 3 tập: train, validation, và test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Chuyển đổi dữ liệu sang tensor PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Sử dụng DataLoader cho việc chia batch dữ liệu
train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=512)

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập xác thực: {X_val.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# ===================== KIẾN TRÚC MÔ HÌNH TỐI ƯU =====================

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Thêm Feed-Forward Network (FFN) với hàm kích hoạt GELU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention + residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        
        # FFN + residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class OptimizedTabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        
        # Input embedding với LayerNorm và GELU để ổn định
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            OptimizedMultiHeadAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classifier head - phức tạp hơn để học các đặc trưng cuối cùng
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(64, 1)
        )
        
        # Khởi tạo trọng số để huấn luyện ổn định hơn
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Embed đầu vào và add sequence dimension
        x = self.embedding(x).unsqueeze(1)
        
        # Pass through transformer
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classify
        return self.classifier(x)

# ===================== THIẾT LẬP HUẤN LUYỆN =====================

# Khởi tạo mô hình
model = OptimizedTabTransformer(
    input_dim=X_train.shape[1],
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    dropout=0.2
)

print(f"Số tham số của mô hình: {sum(p.numel() for p in model.parameters()):,}")

# Use BCEWithLogitsLoss for numerical stability (no need Sigmoid in model)
criterion = nn.BCEWithLogitsLoss()

# Sử dụng AdamW optimizer với weight_decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Sử dụng scheduler để giảm learning rate khi mô hình không cải thiện
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-5
)

# Hàm huấn luyện mỗi epoch
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(pred) > 0.5).float()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Hàm đánh giá toàn bộ mô hình
def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            output = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy().flatten())
            y_prob.extend(output.cpu().numpy().flatten())
            y_pred.extend((output.cpu().numpy() > 0.5).astype(int).flatten())
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    return acc, prec, rec, f1, auc

# ===================== VÒNG LẶP HUẤN LUYỆN =====================

print("Bắt đầu quá trình huấn luyện tối ưu...")
best_val_acc = 0
best_val_f1 = 0
patience_counter = 0
patience = 15

for epoch in range(80):
    # Training
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion)
    
    # Validation
    val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(model, val_dl)
    scheduler.step(val_acc)
    
    if epoch % 5 == 0 or val_acc > best_val_acc:
        print(f"Epoch {epoch+1:2d}: "
              f"Train_Loss={train_loss:.4f}, Train_Acc={train_acc:.4f}, "
              f"Val_Acc={val_acc:.4f}, Val_F1={val_f1:.4f}, Val_AUC={val_auc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_f1': val_f1}, 'best_model.pth')
        print(f"  ✓ Đã lưu mô hình tốt nhất mới: Acc={best_val_acc:.4f}, F1={best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Dừng sớm tại epoch {epoch+1}")
            break

print("\nĐã tải mô hình tốt nhất...")
checkpoint = torch.load('best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print("\n" + "="*60)
print("KẾT QUẢ CUỐI CÙNG:")
print("="*60)

test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_model(model, test_dl)

print(f"Độ chính xác (Accuracy):  {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Độ chính xác dương tính (Precision): {test_prec:.4f}")
print(f"Độ nhạy (Recall):    {test_rec:.4f}")
print(f"Điểm F1 (F1 Score):  {test_f1:.4f}")
print(f"Điểm AUC (AUC):       {test_auc:.4f}")
