import os, cv2, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\manoj\Downloads\final_image\Dataset"
MASK_DIR    = r"C:\Users\manoj\Downloads\final_image\Dataset\Masks"
SAVE_DIR    = "saved_model"
IMG_SIZE    = 128      # ← reduced from 256
BATCH_SIZE  = 32       # ← increased from 8
EPOCHS      = 10       # ← reduced from 25
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED   = (".jpg", ".jpeg", ".png", ".bmp")
os.makedirs(SAVE_DIR, exist_ok=True)
print("Device:", DEVICE)

# ── DATASET ───────────────────────────────────────────────────────────────────
class ForgerySegDataset(Dataset):
    def __init__(self, img_paths, labels, mask_paths, img_tf):
        self.img_paths  = img_paths
        self.labels     = labels
        self.mask_paths = mask_paths
        self.img_tf     = img_tf

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.img_tf(img)

        label = self.labels[idx]
        if label == 1 and self.mask_paths[idx] and os.path.exists(self.mask_paths[idx]):
            msk = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if msk is None:
                msk = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
        elif label == 1:
            msk = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
        else:
            msk = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))
        msk = torch.FloatTensor(msk / 255.0).unsqueeze(0)
        return img, label, msk


# ── MODEL ─────────────────────────────────────────────────────────────────────
class BayarConv2d(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ks=5):
        super().__init__()
        self.kernel = nn.Parameter(torch.rand(out_ch, in_ch, ks, ks))
        self.c = ks // 2

    def forward(self, x):
        k = self.kernel.clone()
        k[:, :, self.c, self.c] = 0
        k[:, :, self.c, self.c] = -k.sum(dim=[2, 3])
        return F.conv2d(x, k, padding=self.c)


class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]]
        f2 = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        f3 = [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]]
        filters = np.array(
            [[[f1,f1,f1]], [[f2,f2,f2]], [[f3,f3,f3]]],
            dtype=np.float32
        ).reshape(3, 3, 5, 5)
        filters[0] /= 4.0
        filters[1] /= 12.0
        filters[2] /= 2.0
        self.register_buffer("w", torch.FloatTensor(filters))

    def forward(self, x):
        return F.conv2d(x, self.w, padding=2)


class ManTraNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.bayar = BayarConv2d()
        self.srm   = SRMConv2d()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # Decoder with skip connections
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.seg_head = nn.Conv2d(64, 1, 1)

        # Classification head
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        noise = torch.cat([x, self.bayar(x), self.srm(x)], dim=1)
        e1    = self.enc1(noise)
        e2    = self.enc2(self.pool1(e1))
        e3    = self.enc3(self.pool2(e2))
        bn    = self.bottleneck(self.pool3(e3))

        # Classification
        cls_out = self.cls_head(self.gap(bn))

        # Segmentation
        d3      = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2      = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1      = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_out = torch.sigmoid(self.seg_head(d1))

        return cls_out, seg_out


# ── LOAD FILE PATHS ───────────────────────────────────────────────────────────
print("\n[1/4] Scanning dataset folders...")
paths, labels, masks = [], [], []

orig_folder = os.path.join(DATASET_DIR, "Original")
if not os.path.isdir(orig_folder):
    raise FileNotFoundError("Folder not found: " + orig_folder)
orig_files = [f for f in os.listdir(orig_folder) if f.lower().endswith(SUPPORTED)]
print("  Original: " + str(len(orig_files)) + " images")
for f in orig_files:
    paths.append(os.path.join(orig_folder, f))
    labels.append(0)
    masks.append(None)

tamp_folder = os.path.join(DATASET_DIR, "Tamper")
if not os.path.isdir(tamp_folder):
    raise FileNotFoundError("Folder not found: " + tamp_folder)
tamp_files = [f for f in os.listdir(tamp_folder) if f.lower().endswith(SUPPORTED)]
print("  Tamper:   " + str(len(tamp_files)) + " images")

mask_found = 0
for f in tamp_files:
    paths.append(os.path.join(tamp_folder, f))
    labels.append(1)
    mask_path = None
    if os.path.isdir(MASK_DIR):
        base = os.path.splitext(f)[0]
        for ext in [".png", ".jpg", ".bmp"]:
            candidate = os.path.join(MASK_DIR, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                mask_found += 1
                break
    masks.append(mask_path)

labels = np.array(labels)
print("  Masks found: " + str(mask_found) + "/" + str(len(tamp_files)))
print("  Total: " + str(len(paths)) + " images")

tr_p, te_p, tr_l, te_l, tr_m, te_m = train_test_split(
    paths, labels, masks, test_size=0.2, random_state=42, stratify=labels
)
print("  Train: " + str(len(tr_p)) + "  |  Test: " + str(len(te_p)))

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = ForgerySegDataset(tr_p, tr_l, tr_m, train_tf)
test_ds  = ForgerySegDataset(te_p, te_l, te_m, val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print("\n[2/4] Building ManTraNet V3...")
model     = ManTraNetV3().to(DEVICE)
total_par = sum(p.numel() for p in model.parameters())
print("  Parameters: " + "{:,}".format(total_par))

counts      = np.bincount(tr_l)
cls_weights = torch.FloatTensor([1.0/counts[0], 1.0/counts[1]]).to(DEVICE)
cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weights)
seg_loss_fn = nn.BCELoss()
optimizer   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
print("\n[3/4] Training for " + str(EPOCHS) + " epochs...")
best_acc   = 0.0
train_accs = []
val_accs   = []

for epoch in range(EPOCHS):
    model.train()
    correct, total, run_loss = 0, 0, 0.0
    for imgs, lbls, msks in train_dl:
        imgs = imgs.to(DEVICE)
        lbls = lbls.long().to(DEVICE)
        msks = msks.to(DEVICE)
        optimizer.zero_grad()
        cls_out, seg_out = model(imgs)
        loss_cls = cls_loss_fn(cls_out, lbls)
        loss_seg = seg_loss_fn(seg_out, msks)
        loss     = loss_cls + 0.5 * loss_seg
        loss.backward()
        optimizer.step()
        preds    = cls_out.argmax(1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
        run_loss += loss.item()
    scheduler.step()
    tr_acc = correct / total * 100

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, lbls, msks in test_dl:
            imgs, lbls = imgs.long().to(DEVICE), lbls.to(DEVICE)
            cls_out, _ = model(imgs)
            preds = cls_out.argmax(1)
            vc   += (preds == lbls).sum().item()
            vt   += lbls.size(0)
    val_acc = vc / vt * 100
    train_accs.append(tr_acc)
    val_accs.append(val_acc)

    tag = ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ManTraNet.pt"))
        tag = "  <-- best saved"

    print("  Epoch " + str(epoch+1).zfill(2) + "/" + str(EPOCHS) +
          "  Train: " + str(round(tr_acc, 1)) + "%" +
          "  Val: "   + str(round(val_acc, 1)) + "%" +
          "  Loss: "  + str(round(run_loss/len(train_dl), 4)) + tag)

# ── PLOT ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_accs, label="Train",      color="#00C8E0", linewidth=2)
plt.plot(range(1, EPOCHS+1), val_accs,   label="Validation", color="#00D4A0", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.title("ManTraNet V3 Training Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curve_v3.png"))
plt.close()

print("\n[4/4] Training complete!")
print("  Best Validation Accuracy: " + str(round(best_acc, 2)) + "%")
print("  Saved: saved_model/ManTraNet.pt")
print("  Saved: saved_model/training_curve_v3.png")
print("\nNow run: python app_v3.py")