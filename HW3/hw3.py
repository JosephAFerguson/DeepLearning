# Joe Ferguson Deep Learning HW2
import os 
import torch
from PIL import Image
import cv2
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math, time, numpy as np
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories for the data
trainImages = "./train/image"
trainMaskImages = "./train/mask"
testImages = "./test/image"
testMaskImages = "./test/mask"

# walk throguh the directories and get the file names
trainImageFiles = []

for root, dirs, files in os.walk(trainImages):
    for file in files:
        if file.endswith(".png"):
            trainImageFiles.append(os.path.join(root, file))

trainMaskFiles = []
for root, dirs, files in os.walk(trainMaskImages):
    for file in files:
        if file.endswith(".png"):
            trainMaskFiles.append(os.path.join(root, file))

testImageFiles = []
for root, dirs, files in os.walk(testImages):
    for file in files:
        if file.endswith(".png"):
            testImageFiles.append(os.path.join(root, file))

testMaskFiles = []
for root, dirs, files in os.walk(testMaskImages):
    for file in files:
        if file.endswith(".png"):
            testMaskFiles.append(os.path.join(root, file))


####----Data Exploration----####

# Q - number of data samples (train,test)
print(f"Number of training images: {len(trainImageFiles)} images, {len(trainMaskFiles)} masks")
print(f"Number of testing images: {len(testImageFiles)} images, {len(testMaskFiles)} masks")

counts = 0
corrupt = []
sizes = Counter()
nonbinary_masks = []


for trainImg, maskImg in zip(trainImageFiles, trainMaskFiles):
    img = Image.open(trainImg)
    sizes[img.size] += 1
    
    m = Image.open(maskImg).convert('L')
    sizes[m.size] += 1
    arr = np.array(m)
    uniq = np.unique(arr)
    
    # not all zeros and ones
    if not set(uniq).issubset({0,255,1}):
        nonbinary_masks.append(maskImg)

    counts += 1

"""def compute_mean_std(trainImageFiles):
    means = []
    stds = []

    for img_path in tqdm(trainImageFiles, desc="Computing mean/std"):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        means.append(np.mean(img, axis=(0,1)))
        stds.append(np.std(img, axis=(0,1)))

    means = np.array(means)
    stds = np.array(stds)

    dataset_mean = means.mean(axis=0)
    dataset_std = stds.mean(axis=0)
    return dataset_mean, dataset_std

dataset_mean, dataset_std = compute_mean_std(trainImageFiles)
print(f"Dataset mean: {dataset_mean}")
print(f"Dataset std: {dataset_std}")"""

print("Unique sizes and counts (width,height):", sizes.most_common()[:10])
# All pngs are (512,512)
#print("Non-binary masks examples:", nonbinary_masks)
# Masks are not grayscale all the time, will need to convert to binary

valImageFiles, finalTestImageFiles, valMaskFiles, finalTestMaskFiles = train_test_split(
    testImageFiles, testMaskFiles, test_size=0.5, random_state=42
)
print(f"Training set: {len(trainImageFiles)} images")
print(f"Validation set: {len(valImageFiles)} images")
print(f"Test set: {len(finalTestImageFiles)} images")

####----Preprocessing----####
class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, size=(512,512), augment=False, amplify_green=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.augment = augment
        self.amplify_green = amplify_green

        # Train dataset mean and std computed earlier
        self.mean = [0.50398916, 0.2740966, 0.16413026]
        self.std  = [0.32702574, 0.17606696, 0.09725136]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Convert to numpy arrays
        img = np.array(img).astype(np.float32) / 255.0

        if self.amplify_green:
            green = img[..., 1]
            green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply((green*255).astype(np.uint8))
            img[..., 1] = green / 255.0

        # Convert grayscale mask to binary 0/1
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        # Normalize image
        img = (img - np.array(self.mean)) / np.array(self.std)
        img = np.transpose(img, (2,0,1))

        return torch.from_numpy(img).float(), torch.from_numpy(mask).unsqueeze(0).float()

train_dataset = RetinaDataset(trainImageFiles, trainMaskFiles)
val_dataset = RetinaDataset(valImageFiles, valMaskFiles)
test_dataset = RetinaDataset(finalTestImageFiles, finalTestMaskFiles)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

def dice_from_probs(probs, target, eps=1e-6):
    """
    probs: float tensor in [0,1], shape (B,1,H,W)
    target: float tensor 0/1, shape (B,1,H,W)
    returns mean dice coefficient across batch (not loss)
    """
    assert probs.shape == target.shape
    intersection = (probs * target).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (denom + eps)
    return dice  # per-sample dice (tensor)

def iou_from_probs(probs, target, thresh=None, eps=1e-6):
    """
    If thresh is None -> compute soft IoU (using probs directly).
    If thresh is a float -> binarize with that threshold then compute IoU.
    Returns per-sample IoU (tensor).
    """
    assert probs.shape == target.shape
    if thresh is not None:
        preds = (probs > thresh).float()
    else:
        preds = probs  # soft IoU
    intersection = (preds * target).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou  # per-sample iou (tensor)

# -------------------------------
# Simple Academic U-Net (Ronneberger 2015)
# -------------------------------
class DoubleConv(nn.Module):
    """(conv => ReLU => conv => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.down3(x4)
        x6 = self.pool3(x5)
        x7 = self.down4(x6)
        x8 = self.pool4(x7)

        # Bottleneck
        x9 = self.bottom(x8)

        # Decoder + skip connections
        x = self.up1(x9)
        x = torch.cat([x7, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x5, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        x = self.out_conv(x)
        return x

model = UNet(in_channels=3, out_channels=1)

def train_model(num_epochs, learning_rate):
    train_losses, val_losses = [], []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss, dice_score, iou_score = 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)     # probabilities in [0,1]

                # per-sample dice/iou tensors
                dice_per_sample = dice_from_probs(probs, masks)   # returns tensor shape (B,)
                iou_per_sample  = iou_from_probs(probs, masks, thresh=0.3)

                # accumulate mean across the mini-batch
                dice_score += dice_per_sample.mean().item()
                iou_score  += iou_per_sample.mean().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_score / len(val_loader)
        avg_iou = iou_score / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"| Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")

    # -------------------------------
    # Plot losses
    # -------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("U-Net Training vs Validation Loss")
    plt.legend()
    plt.show()

    # -------------------------------
    # Save model
    # -------------------------------
    torch.save(model.state_dict(), "unet_brain_mri.pth")
    print("✅ Model saved as unet_brain_mri.pth")

def test_model(model, load=False):
    if load:
        model.load_state_dict(torch.load("unet_brain_mri.pth", map_location=device))
        print("✅ Model loaded from unet_brain_mri.pth")

    model.eval()

    all_dice_scores = []
    all_iou_scores = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            probs = torch.sigmoid(outputs)  # probabilities ∈ [0,1]

            dice_batch = dice_from_probs(probs, masks)
            iou_batch = iou_from_probs(probs, masks)

            all_dice_scores.extend(dice_batch.cpu().numpy())
            all_iou_scores.extend(iou_batch.cpu().numpy())

            # ---------- VISUALIZATION ----------
            if i < 3:
                pred_mask = (probs[0,0].cpu().numpy() > 0.3).astype(np.uint8)
                img = images[0].permute(1,2,0).cpu().numpy()
                gt = masks[0,0].cpu().numpy()

                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.title("Original Image")
                plt.imshow(img)
                plt.axis('off')

                plt.subplot(1,3,2)
                plt.title("Ground Truth Mask")
                plt.imshow(gt, cmap='gray')
                plt.axis('off')

                plt.subplot(1,3,3)
                plt.title("Predicted Mask")
                plt.imshow(pred_mask, cmap='gray')
                plt.axis('off')
                plt.show()

    # ---------- FINAL METRICS ----------
    avg_dice = np.mean(all_dice_scores)
    avg_iou = np.mean(all_iou_scores)

    print(f"✅ Average Dice Score on Test Set: {avg_dice:.4f}")
    print(f"✅ Average IoU Score on Test Set:  {avg_iou:.4f}")

if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1).to(device)
    train_model(num_epochs=50, learning_rate=0.001)
    test_model(model, load=False)