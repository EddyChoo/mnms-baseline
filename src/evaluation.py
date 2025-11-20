import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------

test_root = r"C:\Users\29209\Desktop\Thesis\MEGA\OpenDataset\Testing"
pred_root = r"outputs/unet/predictions"
save_dir  = r"outputs/unet/eval"
os.makedirs(save_dir, exist_ok=True)

# Classes in M&Ms dataset:
# 1 = LV, 2 = RV, 3 = MYO
labels = [1, 2, 3]
class_names = ["LV", "RV", "MYO"]


# ----------------------------
# Dice coefficient (per class)
# ----------------------------
def dice_score(pred, gt, cls):
    pred_bin = (pred == cls).astype(np.int32)
    gt_bin   = (gt == cls).astype(np.int32)

    intersection = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()

    if denom == 0:
        return np.nan  # skip empty cases
    return 2 * intersection / denom


# ----------------------------
# Evaluation loop
# ----------------------------
results = []
cases = sorted(os.listdir(test_root))

print("\n[INFO] Starting evaluation...\n")

for case in cases:
    case_dir = os.path.join(test_root, case)
    img_files = [f for f in os.listdir(case_dir) if f.endswith("_sa_gt.nii.gz")]

    if len(img_files) == 0:
        print(f"[WARNING] No GT mask for {case}")
        continue

    gt_path = os.path.join(case_dir, img_files[0])
    pred_path = os.path.join(pred_root, f"{case}_pred.nii.gz")

    if not os.path.exists(pred_path):
        print(f"[WARNING] Prediction missing: {pred_path}")
        continue

    print(f"[INFO] Evaluating case {case}...")

    # Load 4D ground truth and prediction
    gt = nib.load(gt_path).get_fdata().astype(np.int16)    # (H,W,S,T)
    pred = nib.load(pred_path).get_fdata().astype(np.int16)

    case_scores = []

    for cls in labels:
        dsc = dice_score(pred, gt, cls)
        case_scores.append(dsc)

    mean_dice = np.nanmean(case_scores)

    results.append({
        "Case": case,
        "LV Dice": case_scores[0],
        "RV Dice": case_scores[1],
        "MYO Dice": case_scores[2],
        "Mean Dice": mean_dice
    })

    print(f"  LV={case_scores[0]:.3f}, RV={case_scores[1]:.3f}, MYO={case_scores[2]:.3f}, "
          f"Mean={mean_dice:.3f}")


# ----------------------------
# Save results
# ----------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(save_dir, "unet_dice_results.csv")
df.to_csv(csv_path, index=False)

print(f"\n[INFO] Results saved to: {csv_path}")
print(df)

# ----------------------------
# Plot bar chart (Mean Dice)
# ----------------------------
plt.figure(figsize=(10,5))
plt.bar(df["Case"], df["Mean Dice"], color="steelblue")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.title("UNet Mean Dice per Case")
plt.ylabel("Mean Dice")
plt.tight_layout()

plot_path = os.path.join(save_dir, "unet_mean_dice.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"[INFO] Bar plot saved: {plot_path}")

print("\n[INFO] Evaluation complete!")
