import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
# Input Data (Prepared in previous step)
DATA_FILE = Path('/home/francesco/Francesco/isaacsim_projects/AdversarialGhosting/QualityAssessment/step1/model/surrogate_training_data_F6_v2.pt')
# Output Model File
# OUTPUT_MODEL = DATA_FILE.parent / 'surrogate_model_F4.pth'
OUTPUT_MODEL = DATA_FILE.parent / 'surrogate_model_F6_v2.pth'
# Output Loss Plot
PLOT_FILE = DATA_FILE.parent / 'surrogate_training_loss_F6_v2.png'

# 1. Define the Model Architecture g_ell(theta)
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=128):
        super(SurrogateModel, self).__init__()
        # We use an "Enhanced MLP" for better regression accuracy
        self.net = nn.Sequential(
            # Layer 1: Input (x,y) -> High dim feature space
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32), # Stabilizes training
            nn.GELU(),           # Smooth activation (better for regression than ReLU)
            
            # Layer 2: Deep processing
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            # Layer 3: Output the Feature Vector Delta Phi
            nn.Linear(128, output_dim) 
        )
        
    def forward(self, theta):
        return self.net(theta)

def train_and_save():
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    
    # --- Hardware Device ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Dataset D ---
    print(f"Loading data from {DATA_FILE.name}...")
    data = torch.load(DATA_FILE)
    Theta = data['X']        # Inputs: theta_j (Position)
    DeltaPhi = data['Y']     # Targets: Delta phi_ell (Feature Perturbation)

    # Ensure shapes are correct (N, 2) and (N, 128)
    if Theta.dim() == 3: Theta = Theta.squeeze(1)

    # Split Train (80%) / Validation (20%)
    N = len(Theta)
    n_train = int(N * 0.8)
    indices = torch.randperm(N)
    
    Theta_train, Theta_val = Theta[indices[:n_train]], Theta[indices[n_train:]]
    DeltaPhi_train, DeltaPhi_val = DeltaPhi[indices[:n_train]], DeltaPhi[indices[n_train:]]

    print(f"Training on {n_train} samples, Validating on {N - n_train} samples.")

    # --- Setup DataLoader for Mini-Batching ---
    batch_size = 64
    train_dataset = TensorDataset(Theta_train, DeltaPhi_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Setup Model & Optimization ---
    # FIX: Dynamically read dimensions directly from the tensors!
    actual_input_dim = Theta.shape[1]      # Will be 3 (X, Y, Angle)
    actual_output_dim = DeltaPhi.shape[1]  # The feature vector size for this specific layer

    print(f"Initializing Surrogate Model: Input={actual_input_dim}, Output={actual_output_dim}")
    model = SurrogateModel(input_dim=actual_input_dim, output_dim=actual_output_dim).to(device)
    
    # Regression Loss: L = Sum || g(theta) - DeltaPhi ||^2
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # ADD THIS: Shrinks LR by half if validation loss doesn't improve for 100 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)


    # --- Training Loop ---
    epochs = 2000
    train_losses = []
    val_losses = []

    # FIX: Send the validation data to the device BEFORE the loop starts
    Theta_val = Theta_val.to(device)
    DeltaPhi_val = DeltaPhi_val.to(device)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Mini-Batch Iteration
        for batch_theta, batch_delta in train_loader:
            batch_theta = batch_theta.to(device)
            batch_delta = batch_delta.to(device)
            optimizer.zero_grad()

            # 1. Forward Pass
            prediction = model(batch_theta)
            # 2. Compute Loss
            loss = criterion(prediction, batch_delta)
        
            # 3. Backward Pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_theta.size(0)
        
        avg_train_loss = epoch_loss / n_train
        train_losses.append(avg_train_loss)

        # Validation (Check generalization)
        model.eval()
        with torch.no_grad():
            val_pred = model(Theta_val)
            val_loss = criterion(val_pred, DeltaPhi_val)
            val_losses.append(val_loss.item())

        scheduler.step(val_loss)
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f} | Val Loss = {val_loss.item():.6f}")

    # --- SAVE THE MODEL ---
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\n✅ Model successfully saved to: {OUTPUT_MODEL}")

    # --- VERIFICATION STEP (Point 3 Requirement) ---
    print("\n--- Verification: Prediction Quality ---")
    model.eval()
    with torch.no_grad():
        # Compute Residuals on Validation Set
        g_theta = model(Theta_val)
        residuals = g_theta - DeltaPhi_val
        
        # Compute L2 Norm per sample
        norms = torch.norm(residuals, p=2, dim=1)
        
        avg_error = norms.mean().item()
        print(f"Average Prediction Error ||g(theta) - DeltaPhi||: {avg_error:.4f}")
        
        if avg_error < 1.0:
            print("Status: EXCELLENT (Model captures the perturbation accurately)")
        elif avg_error < 5.0:
            print("Status: GOOD (Model captures the general trend)")
        else:
            print("Status: POOR (Model needs more data or training)")

    # Plot Loss Curve
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title(r'Regression Loss $\mathcal{L} = \sum \|g_\ell(\theta) - \Delta\phi_\ell(\theta)\|^2$')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_FILE)
    print(f"Loss plot saved to: {PLOT_FILE}")

if __name__ == '__main__':
    train_and_save()