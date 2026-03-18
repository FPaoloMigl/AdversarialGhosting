import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
DATA_FILE = Path('AdversarialGhosting/QualityAssessment/step3_angle_decompose/model/surrogate_training_data_F6_v3_sincos.pt')
OUTPUT_MODEL = DATA_FILE.parent / 'surrogate_model_F6_v3_sincos.pth'
PLOT_FILE = DATA_FILE.parent / 'surrogate_training_loss_F6_v3_sincos.png'


# === MODEL (input_dim=4 for [x, y, sin(a), cos(a)]) ===
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, output_dim),
        )
        
    def forward(self, theta):
        return self.net(theta)


def train_and_save():
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Dataset ---
    print(f"Loading data from {DATA_FILE.name}...")
    data = torch.load(DATA_FILE)
    Theta = data['X']        # [N, 4] — [x_norm, y_norm, sin(a), cos(a)]
    DeltaPhi = data['Y']     # [N, 128] — Top-K Mean Delta Phi
    
    print(f"Input encoding: {data.get('encoding', 'unknown')}")
    print(f"Top-K: {data.get('top_k', 'unknown')}")

    if Theta.dim() == 3:
        Theta = Theta.squeeze(1)

    # --- Split Train/Val ---
    N = len(Theta)
    n_train = int(N * 0.8)
    
    torch.manual_seed(42)
    indices = torch.randperm(N)
    
    Theta_train, Theta_val = Theta[indices[:n_train]], Theta[indices[n_train:]]
    DeltaPhi_train, DeltaPhi_val = DeltaPhi[indices[:n_train]], DeltaPhi[indices[n_train:]]

    print(f"Training on {n_train} samples, Validating on {N - n_train} samples.")

    # --- DataLoader ---
    batch_size = 64
    train_dataset = TensorDataset(Theta_train, DeltaPhi_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Model Setup ---
    actual_input_dim = Theta.shape[1]      # 4
    actual_output_dim = DeltaPhi.shape[1]  # 128

    print(f"Initializing Surrogate Model: Input={actual_input_dim}, Output={actual_output_dim}")
    model = SurrogateModel(input_dim=actual_input_dim, output_dim=actual_output_dim).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    # --- Training Loop ---
    epochs = 3000
    train_losses = []
    val_losses = []

    Theta_val = Theta_val.to(device)
    DeltaPhi_val = DeltaPhi_val.to(device)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_theta, batch_delta in train_loader:
            batch_theta = batch_theta.to(device)
            batch_delta = batch_delta.to(device)
            optimizer.zero_grad()

            prediction = model(batch_theta)
            loss = criterion(prediction, batch_delta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_theta.size(0)
        
        avg_train_loss = epoch_loss / n_train
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(Theta_val)
            val_loss = criterion(val_pred, DeltaPhi_val)
            val_losses.append(val_loss.item())

        scheduler.step(val_loss)
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f} | Val Loss = {val_loss.item():.6f}")

    # --- Save ---
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\nModel saved to: {OUTPUT_MODEL}")

    # --- Verification ---
    print("\n--- Verification: Prediction Quality ---")
    model.eval()
    with torch.no_grad():
        g_theta = model(Theta_val)
        norms = torch.norm(g_theta - DeltaPhi_val, p=2, dim=1)
        avg_error = norms.mean().item()
        print(f"Average Prediction Error ||g(theta) - DeltaPhi||: {avg_error:.4f}")
        
        if avg_error < 1.0:
            print("Status: EXCELLENT")
        elif avg_error < 5.0:
            print("Status: GOOD")
        else:
            print("Status: POOR")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title(r'Surrogate Training Loss (sin/cos + Top-K)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_FILE)
    print(f"Loss plot saved to: {PLOT_FILE}")


if __name__ == '__main__':
    train_and_save()