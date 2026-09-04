import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import Sentinel2SRDataset

from model_edsr_improved import EDSRImproved


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"

EPOCHS = 20

BATCH_SIZE = 4

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "checkpoints"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)

print("DEVICE:", device)

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

dataset = Sentinel2SRDataset(
    DATASET_ROOT,
    split="train"
)

print("TRAIN dataset:")

print("Training samples:", len(dataset))


train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Batches per epoch:", len(train_loader))


# ============================================================
# MODEL
# ============================================================

model = EDSRImproved(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
    scale=4
)

model = model.to(device)

print("Improved EDSR model created successfully!")


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(train_loader):

        lr = lr.to(device)

        hr = hr.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        sr = model(lr)

        # Keep output in valid range
        sr = torch.clamp(
            sr,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            sr,
            hr
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        running_loss /
        len(train_loader)
    )

    print()

    print("=" * 50)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed"
    )

    print(
        f"Average Loss: {average_loss:.6f}"
    )

    print("=" * 50)

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"edsr_improved_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        "Saved:",
        checkpoint_path
    )

    print()


# ============================================================
# COMPLETED
# ============================================================

print("=" * 60)

print("IMPROVED EDSR TRAINING COMPLETED!")

print("=" * 60)