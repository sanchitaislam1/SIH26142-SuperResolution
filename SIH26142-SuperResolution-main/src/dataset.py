from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Sentinel2SRDataset(Dataset):
    """
    Dataset loader for Sentinel-2 super-resolution.

    LR:
        4 x 32 x 32
        Channels: B04, B03, B02, B08

    HR:
        4 x 128 x 128
        Channels: B04, B03, B02, B08
    """

    def __init__(self, dataset_root, split="train"):
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.lr_dir = self.dataset_root / split / "lr"
        self.hr_dir = self.dataset_root / split / "hr"

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR folder not found: {self.lr_dir}")

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")

        # Get all LR files
        self.lr_files = sorted(self.lr_dir.glob("*.npy"))

        # Keep only files that have a matching HR file
        self.pairs = []

        for lr_file in self.lr_files:
            hr_file = self.hr_dir / lr_file.name

            if hr_file.exists():
                self.pairs.append((lr_file, hr_file))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching LR-HR pairs found in {split}."
            )

        print(f"{split.upper()} dataset:")
        print(f"  LR files found: {len(self.lr_files)}")
        print(f"  Matching pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]

        # Load NumPy arrays
        lr = np.load(lr_path).astype(np.float32)
        hr = np.load(hr_path).astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return lr, hr


if __name__ == "__main__":

    # CHANGE THIS PATH if your dataset is somewhere else
    DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"

    # Load training dataset
    dataset = Sentinel2SRDataset(
        DATASET_ROOT,
        split="train"
    )

    # Read the first LR-HR pair
    lr, hr = dataset[0]

    print("\n===== FIRST PAIR =====")
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)

    print("\n===== DATA TYPE =====")
    print("LR dtype:", lr.dtype)
    print("HR dtype:", hr.dtype)

    print("\n===== VALUE RANGE =====")
    print("LR min:", lr.min().item())
    print("LR max:", lr.max().item())
    print("HR min:", hr.min().item())
    print("HR max:", hr.max().item())

    print("\nDataset loader is working!")