import numpy as np

# Change this path only if your dataset is stored somewhere else
lr_path = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset\train\lr\patch_0218.npy"
hr_path = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset\train\hr\patch_0218.npy"

# Load the LR and HR patches
lr = np.load(lr_path)
hr = np.load(hr_path)

print("===== LOW-RESOLUTION (LR) =====")
print("Shape:", lr.shape)
print("Data type:", lr.dtype)
print("Minimum value:", lr.min())
print("Maximum value:", lr.max())

print("\n===== HIGH-RESOLUTION (HR) =====")
print("Shape:", hr.shape)
print("Data type:", hr.dtype)
print("Minimum value:", hr.min())
print("Maximum value:", hr.max())

print("\n===== SIZE COMPARISON =====")
print("LR total values:", lr.size)
print("HR total values:", hr.size)