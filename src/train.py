import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Sentinel2SRDataset
from model import EDSR


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"

# Smaller/faster settings for CPU
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE:", DEVICE)
print("=" * 60)


# ============================================================
# DATASET
# ============================================================

train_dataset = Sentinel2SRDataset(
    DATASET_ROOT,
    split="train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Batches per epoch:", len(train_loader))


# ============================================================
# MODEL
# ============================================================

model = EDSR(
    in_channels=4,
    out_channels=4,

    # Smaller model = much faster CPU training
    features=32,
    num_blocks=4,

    scale=4
)

model = model.to(DEVICE)

print("\nModel created successfully.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT FOLDER
# ============================================================

os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(train_loader):

        # Move data to CPU/GPU
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Generate super-resolved image
        sr = model(lr)

        # Calculate loss
        loss = criterion(sr, hr)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Progress
        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    # Average loss
    average_loss = total_loss / len(train_loader)

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS} completed")
    print(f"Average Loss: {average_loss:.6f}")
    print("=" * 60)

    # Save checkpoint
    checkpoint_path = (
        f"checkpoints/edsr_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print("Saved:", checkpoint_path)
    print()


print("TRAINING COMPLETED!")