import os
import glob
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import cv2

# ----------------------------
# Config (SAFE VALUES)
# ----------------------------
TARGET_H = 320
TARGET_W = 320
IMG_DTYPE = torch.float16
LBL_DTYPE = torch.int64

BASE_DIR = r"C:\Users\29209\Desktop\Thesis\MEGA\OpenDataset"
OUT_DIR  = r"C:\Users\29209\Desktop\Thesis\MEGA\Cases_320"
SPLITS   = ["Training", "Validation", "Testing"]


def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)


def resize_slice(img2d, H=320, W=320):
    img2d = img2d.astype(np.float32)
    resized = cv2.resize(img2d, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized


def resize_mask(mask2d, H=320, W=320):
    mask2d = mask2d.astype(np.uint8)
    resized = cv2.resize(mask2d, (W, H), interpolation=cv2.INTER_NEAREST)
    return resized


def preprocess_case(img_path, out_path):
    gt_path = img_path.replace("_sa.nii.gz", "_sa_gt.nii.gz")
    has_gt = os.path.exists(gt_path)

    img4d = nib.load(img_path).get_fdata().astype(np.float32)
    H, W, S, T = img4d.shape

    # normalize entire case
    img4d = (img4d - img4d.mean()) / (img4d.std() + 1e-6)

    out_img = np.zeros((S, T, TARGET_H, TARGET_W), dtype=np.float16)
    out_lbl = np.zeros((S, T, TARGET_H, TARGET_W), dtype=np.uint8) if has_gt else None

    if has_gt:
        lbl4d = nib.load(gt_path).get_fdata().astype(np.uint8)

    # resize all slices
    for s in range(S):
        for t in range(T):
            out_img[s, t] = resize_slice(img4d[:, :, s, t])
            if has_gt:
                out_lbl[s, t] = resize_mask(lbl4d[:, :, s, t])

    case_id = os.path.basename(img_path).replace("_sa.nii.gz", "")

    torch.save(
        {
            "image": torch.tensor(out_img, dtype=IMG_DTYPE),  # (S, T, H, W)
            "label": torch.tensor(out_lbl, dtype=LBL_DTYPE) if has_gt else None,
            "case": case_id,
        },
        out_path,
    )


def preprocess_split(split_name):
    split_in  = os.path.join(BASE_DIR, split_name)
    split_out = os.path.join(OUT_DIR, split_name)
    ensure(split_out)

    # Support Training/Labeled + Training/Unlabeled
    pattern1 = os.path.join(split_in, "Labeled", "*", "*_sa.nii.gz")
    pattern2 = os.path.join(split_in, "Unlabeled", "*", "*_sa.nii.gz")

    img_paths = sorted(glob.glob(pattern1) + glob.glob(pattern2))

    # For Validation + Testing, they don't have Labeled/Unlabeled
    if len(img_paths) == 0:
        img_paths = sorted(glob.glob(os.path.join(split_in, "*", "*_sa.nii.gz")))

    print(f"\n[INFO] {split_name} → {len(img_paths)} cases")

    for idx, img_path in enumerate(img_paths):
        case_id = os.path.basename(img_path).replace("_sa.nii.gz", "")
        out_file = os.path.join(split_out, case_id + ".pt")

        if os.path.exists(out_file):
            print(f"  ({idx+1}) Skip {case_id}")
            continue

        print(f"  ({idx+1}) Preprocess {case_id} ...")
        preprocess_case(img_path, out_file)

    print(f"[OK] Finished: {split_name}")


if __name__ == "__main__":
    ensure(OUT_DIR)
    for sp in SPLITS:
        preprocess_split(sp)
    print("\n=== ALL DONE ===")
