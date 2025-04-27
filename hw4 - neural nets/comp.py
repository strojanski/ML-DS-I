import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# 1) LOAD & PREPARE DATA
fn = "image1-competition.hdf5"
with h5py.File(fn, "r") as f:
    data = f["data"][()]         # shape: (H=934, W=1300, C=187)
    classes = f["classes"][()]   # shape: (934, 1300)

H, W, C = data.shape
flat_data = data.reshape(-1, C)      # (934*1300, 187)
flat_labels = classes.reshape(-1)    # (934*1300,)

# 2) MASK OUT UNANNOTATED
mask = flat_labels != -1
X = flat_data[mask]      # (N_valid, 187)
y = flat_labels[mask]    # (N_valid,)

# 3) STANDARDIZE SPECTRA
scaler = StandardScaler()
X = scaler.fit_transform(X)  # still shape (N_valid, 187)

# 4) DATASET & DATALOADER
class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

ds = SpectralDataset(X, y)
loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=2)

# 5) SIMPLE MLP
class MLP(nn.Module):
    def __init__(self, in_dim=187, hidden=[128,64], n_classes=6):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    # 6) TRAIN
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        n = len(ds)
        print(f"Epoch {epoch+1}: Loss = {total_loss/n:.4f}, Acc = {correct/n:.4f}")

    # 7) INFERENCE ON SUBSET
    sub = data[265:465, 360:660]         # shape (200,300,187)
    sub_flat = sub.reshape(-1, C)
    sub_flat = scaler.transform(sub_flat)
    sub_tensor = torch.from_numpy(sub_flat).float().to(device)
    model.eval()
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(sub_tensor)                # raw outputs, shape (batch_size, n_classes)
        preds  = F.softmax(logits, dim=1)         # convert to probabilities
        preds  = preds.cpu().numpy()              # now a NumPy array of probabilities

        
    preds = preds.astype(np.float32)
    preds = preds.reshape((200,300,6))
    print(preds)
    print("Predictions on subset:", preds.shape)
    np.save("predictions.npy", preds)  # Save predictions for later use
