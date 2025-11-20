import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import AsDiscrete
from monai.networks.nets import UNet

# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_root = r"C:\Users\29209\Desktop\Thesis\MEGA\OpenDataset\Testing"
model_ckpt = r"outputs/unet/checkpoints/best_model.pth"
save_dir   = r"outputs/unet/predictions"

os.makedirs(save_dir, exist_ok=True)

num_classes = 4

# ----------------------------
# Load model
# ----------------------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

print(f"[INFO] Loading checkpoint: {model_ckpt}")
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.eval()

post_pred = AsDiscrete(argmax=True)


# ----------------------------
# Run inference on each case
# ----------------------------
cases = sorted(os.listdir(test_root))

for case in cases:
    case_dir = os.path.join(test_root, case)
    nii_files = [f for f in os.listdir(case_dir) if f.endswith("_sa.nii.gz")]

    if len(nii_files) == 0:
        print(f"[WARNING] No MRI found for {case}")
        continue

    img_path = os.path.join(case_dir, nii_files[0])
    print(f"\n[INFO] Predicting case {case}")

    # Load 4D cine MRI
    img_nib = nib.load(img_path)
    img_4d  = img_nib.get_fdata().astype(np.float32)

    H, W, S, T = img_4d.shape
    print(f"[INFO] Image shape: {img_4d.shape}")

    # Output prediction volume
    pred_volume = np.zeros((H, W, S, T), dtype=np.int16)

    with torch.no_grad():
        for s in range(S):
            for t in range(T):

                # Extract 2D slice
                img2d = img_4d[:, :, s, t]

                # Normalize like training
                img2d = (img2d - img2d.mean()) / (img2d.std() + 1e-8)

                # Convert to tensor: (1,1,H,W)
                x = torch.tensor(img2d).unsqueeze(0).unsqueeze(0).float().to(device)

                # UNet forward pass
                y = model(x)
                y = post_pred(y).cpu().numpy()[0]    # (H,W)

                pred_volume[:, :, s, t] = y

    # Save prediction
    out_path = os.path.join(save_dir, f"{case}_pred.nii.gz")
    pred_nii = nib.Nifti1Image(pred_volume, affine=img_nib.affine)
    nib.save(pred_nii, out_path)

    print(f"[INFO] Saved prediction → {out_path}")

print("\nPrediction complete!")
