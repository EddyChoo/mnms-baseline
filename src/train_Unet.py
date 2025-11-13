import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm

from src.dataset import MnmsDataset


# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root = r"C:\Users\29209\Desktop\Thesis\MEGA\OpenDataset\Training\Labeled"
val_root   = r"C:\Users\29209\Desktop\Thesis\MEGA\OpenDataset\Validation"

batch_size = 2
num_epochs = 50
learning_rate = 1e-4
num_classes = 4  # Background, LV, RV, MYO

# ----------------------------
# Output directories (automatic creation)
# ----------------------------
model_name = "unet"  # Change to "dynunet" for your other script

base_dir = os.path.join("outputs", model_name)
ckpt_dir = os.path.join(base_dir, "checkpoints")
log_dir  = os.path.join(base_dir, "logs")
pred_dir = os.path.join(base_dir, "predictions")

for d in [ckpt_dir, log_dir, pred_dir]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Outputs will be saved under: {base_dir}")


# ----------------------------
# Load datasets
# ----------------------------
train_dataset = MnmsDataset(train_root)
val_dataset   = MnmsDataset(val_root)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)


# ----------------------------
# Define model, loss, optimizer
# ----------------------------
model = UNet(
    spatial_dims=2,       # Treat slices as 2D images
    in_channels=1,        # Grayscale MRI
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

criterion = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = AsDiscrete(argmax=True)
post_label = AsDiscrete(to_onehot=num_classes)


# ----------------------------
# Training loop
# ----------------------------
best_dice = 0.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for batch in pbar:
        imgs = batch["image"].to(device)       # (B,1,H,W,D)
        lbls = batch["label"].to(device)       # (B,H,W,D)
        imgs = imgs.squeeze(-1)                # Take 2D slices
        lbls = lbls.squeeze(-1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # ------------------------
    # Validation
    # ------------------------
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            imgs = imgs.squeeze(-1)
            lbls = lbls.squeeze(-1)

            outputs = model(imgs)
            outputs = post_pred(outputs)
            lbls_onehot = post_label(lbls)
            dice_metric(y_pred=outputs, y=lbls_onehot)

        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Loss: {epoch_loss/len(train_loader):.4f}, Dice: {mean_dice:.4f}")

    # Save best model
    if mean_dice > best_dice:
        best_dice = mean_dice
        ckpt_path = os.path.join(ckpt_dir, f"best_model_epoch{epoch+1}_dice{best_dice:.3f}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved best model to {ckpt_path}")

print("Training complete. Best mean Dice:", best_dice)
