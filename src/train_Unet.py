import os
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from src.dataset import CaseDataset

def main():

    # ----------------------------
    # Configuration
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True

    train_dataset = CaseDataset(r"C:\Users\29209\Desktop\Thesis\MEGA\Cases_320\Training")
    val_dataset = CaseDataset(r"C:\Users\29209\Desktop\Thesis\MEGA\Cases_320\Validation")

    batch_size = 4
    num_epochs = 5
    learning_rate = 1e-4
    num_classes = 4

    # ----------------------------
    # Output directories
    # ----------------------------
    model_name = "unet"

    base_dir = os.path.join("outputs", model_name)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    log_dir  = os.path.join(base_dir, "logs")
    pred_dir = os.path.join(base_dir, "predictions")

    for d in [ckpt_dir, log_dir, pred_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Outputs will be saved under: {base_dir}")

    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] DataLoader workers: {num_workers}")

    # ----------------------------
    # Define model
    # ----------------------------
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred  = AsDiscrete(argmax=True)
    post_label = AsDiscrete(to_onehot=num_classes)

    scaler = GradScaler(device.type)

    print("💡 Device in use:", device)
    print("💡 Model is on:", next(model.parameters()).device)

    # ----------------------------
    # Training loop
    # ----------------------------
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in pbar:
            imgs = batch["image"].to(device)         # (B, 1, H, W)
            lbls = batch["label"].unsqueeze(1).to(device).long()  # (B, 1, H, W)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(imgs)               # (B, 4, H, W)
                loss = criterion(outputs, lbls)     # labels are integer, no channel dim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device).long()
                # FP16 inference
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(imgs)                      # (1,4,H,W)

                pred = torch.argmax(logits, dim=1)  # (B,H,W)

                # ---- 2) one-hot for pred & gt ----
                preds_onehot = torch.nn.functional.one_hot(pred, num_classes=num_classes)
                preds_onehot = preds_onehot.permute(0, 3, 1, 2).float()  # (B,4,H,W)

                gt_onehot = torch.nn.functional.one_hot(lbls, num_classes=num_classes)
                gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()  # (B,4,H,W)

                dice_metric(y_pred=preds_onehot, y=gt_onehot)

            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Loss: {epoch_loss/len(train_loader):.4f}, Dice: {mean_dice:.4f}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = os.path.join(
                ckpt_dir, f"best_model_epoch{epoch+1}_dice{best_dice:.3f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved best model to {ckpt_path}")

    print("Training complete. Best mean Dice:", best_dice)


# -----------------------------------------------------------------
# REQUIRED FOR WINDOWS multiprocessing
# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
