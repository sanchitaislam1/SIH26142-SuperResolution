import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import Sentinel2SRDataset

from model_edsr_baseline import EDSRBaseline


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = (
    r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset"
    r"\content\dataset"
)

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

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "WARNING: CUDA is not available."
    )

    print(
        "Training will run on CPU and may be very slow."
    )


# ============================================================
# DATASET
# ============================================================

dataset = Sentinel2SRDataset(
    DATASET_ROOT,
    split="train"
)

print()
print("=" * 60)
print("TRAIN DATASET")
print("=" * 60)

print(
    "Training samples:",
    len(dataset)
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(
    "Batches per epoch:",
    len(train_loader)
)


# ============================================================
# MODEL
# ============================================================

model = EDSRBaseline(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
    scale=4
)

model = model.to(device)


print()
print("=" * 60)
print("MODEL")
print("=" * 60)

print(
    "Standard EDSR baseline created successfully!"
)

print(
    "Input channels : 4"
)

print(
    "Output channels: 4"
)

print(
    "Features       : 64"
)

print(
    "Residual blocks: 8"
)

print(
    "Scale          : 4"
)


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

print()
print("=" * 60)
print("STARTING BASELINE TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        lr = lr.to(device)

        hr = hr.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        sr = model(lr)


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


        # ----------------------------------------------------
        # Store loss
        # ----------------------------------------------------

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


    # ========================================================
    # AVERAGE LOSS
    # ========================================================

    average_loss = (
        running_loss /
        len(train_loader)
    )


    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed"
    )

    print(
        f"Average Loss: {average_loss:.6f}"
    )

    print("=" * 60)


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"edsr_baseline_epoch_{epoch + 1}.pth"
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

print()
print("=" * 60)
print("BASELINE TRAINING COMPLETED!")
print("=" * 60)