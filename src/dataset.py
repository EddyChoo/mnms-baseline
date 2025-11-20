import os
import glob
import torch
from torch.utils.data import Dataset


class CaseDataset(Dataset):
    """
    Loads entire cases (S x T x H x W) from .pt,
    yields (image_slice, mask_slice)
    """

    def __init__(self, root_dir):
        self.case_paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.all_cases = []
        self.index_map = []

        for case_idx, path in enumerate(self.case_paths):
            data = torch.load(path, map_location="cpu")
            self.all_cases.append(data)
            S, T, _, _ = data["image"].shape
            for s in range(S):
                for t in range(T):
                    self.index_map.append((case_idx, s, t))

        print(f"[Dataset] Total slices: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        case_idx, s, t = self.index_map[idx]

        data = self.all_cases[case_idx]
        img = data["image"][s, t].unsqueeze(0).float()  # (1, H, W)
        lbl = data["label"][s, t].long() if data["label"] is not None else None

        return {
            "image": img,
            "label": lbl,
            "case": data["case"]
        }
