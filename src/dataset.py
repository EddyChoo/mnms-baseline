import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class MnmsDataset(Dataset):
    """
    Custom dataset class for the M&Ms cardiac MRI dataset.
    It automatically detects labeled or unlabeled samples based on file availability.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset split (e.g., Training/Labeled, Validation, Testing)
            transform (callable, optional): Optional transform to be applied to each sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all *_sa.nii.gz files
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*_sa.nii.gz")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz images found in {root_dir}")

        # Detect corresponding ground truth if available
        self.has_gt = []
        self.gt_paths = []
        for img_path in self.image_paths:
            base = img_path.replace("_sa.nii.gz", "_sa_gt.nii.gz")
            if os.path.exists(base):
                self.has_gt.append(True)
                self.gt_paths.append(base)
            else:
                self.has_gt.append(False)
                self.gt_paths.append(None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load MRI image
        img_path = self.image_paths[idx]
        image = nib.load(img_path).get_fdata().astype(np.float32)

        # Normalize to zero mean, unit variance
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)

        # Convert to tensor and add channel dimension
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W, D) → (1, …)

        # Load ground truth if available
        gt_path = self.gt_paths[idx]
        if gt_path is not None:
            label = nib.load(gt_path).get_fdata().astype(np.int64)
            label = torch.tensor(label)
        else:
            label = None

        if self.transform:
            image, label = self.transform(image, label)

        sample = {"image": image, "label": label, "path": img_path}
        return sample
