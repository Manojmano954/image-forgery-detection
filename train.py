import os, cv2, numpy as np, pickle, matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\manoj\Downloads\final_image\Dataset"
SAVE_DIR    = "saved_model"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
SUPPORTED   = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Collect file paths ─────────────────────────────────────────────────────
print("[1/5] Scanning dataset folders...")
categories = {"Original": 0, "Tamper": 1}
all_paths, all_labels = [], []

for label_name, label_idx in categories.items():
    folder = os.path.join(DATASET_DIR, label_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError("Folder not found: " + folder)
    files = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED)]
    print("  " + label_name + ": " + str(len(files)) + " images")
    for fname in files:
        all_paths.append(os.path.join(folder, fname))
        all_labels.append(label_idx)

all_labels = np.array(all_labels)
print("  Total: " + str(len(all_paths)) + " images")

# ── 2. Build EfficientNetB0 ───────────────────────────────────────────────────
print("[2/5] Building EfficientNetB0 feature extractor...")
base = EfficientNetB0(input_shape=(224,224,3), include_top=False, pooling="avg", weights="imagenet")
base.trainable = False
fe = Model(inputs=base.input, outputs=base.output)
fe.save(os.path.join(SAVE_DIR, "efficientnet_extractor.h5"))
print("  Feature vector: " + str(base.output_shape[-1]) + " dims")

# ── 3. Extract features batch by batch ───────────────────────────────────────
print("[3/5] Extracting features in batches...")
all_features = []
n = len(all_paths)

for start in range(0, n, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n)
    batch = []
    for path in all_paths[start:end]:
        img = cv2.imread(path)
        if img is None:
            batch.append(np.zeros((224, 224, 3), dtype="float32"))
            continue
        img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
        batch.append(img.astype("float32"))
    batch_arr = np.array(batch, dtype="float32")
    features  = fe.predict(batch_arr, verbose=0)
    all_features.append(features)
    del batch, batch_arr
    if (start // BATCH_SIZE) % 50 == 0:
        done = end / n * 100
        print("  Progress: " + str(end) + "/" + str(n) + " (" + str(round(done,1)) + "%)")

all_features = np.vstack(all_features)
print("  Features shape: " + str(all_features.shape))

# ── 4. Scale features ─────────────────────────────────────────────────────────
print("[4/5] Scaling features...")
scaler  = StandardScaler()
feats_s = scaler.fit_transform(all_features)
pickle.dump(scaler, open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb"))
del all_features

# ── 5. Train LinearSVC ────────────────────────────────────────────────────────
print("[5/5] Training LinearSVC classifier...")
Xtr, Xte, ytr, yte = train_test_split(feats_s, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
print("  Train: " + str(len(Xtr)) + "  |  Test: " + str(len(Xte)))

classes = np.unique(all_labels)
cw      = compute_class_weight("balanced", classes=classes, y=all_labels)
wd      = dict(zip(classes.astype(int), cw))
print("  Class weights: " + str(wd))

svm_base = LinearSVC(C=1.0, max_iter=2000, class_weight=wd)
svm      = CalibratedClassifierCV(svm_base)
svm.fit(Xtr, ytr)

ypred = svm.predict(Xte)
acc   = accuracy_score(yte, ypred) * 100
print("  Test Accuracy: " + str(round(acc, 2)) + "%")
print(classification_report(yte, ypred, target_names=["Original", "Tamper"]))

cm = confusion_matrix(yte, ypred)
ConfusionMatrixDisplay(cm, display_labels=["Original", "Tamper"]).plot(cmap="Blues")
plt.title("Confusion Matrix - V2 EfficientNetB0")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

pickle.dump(svm, open(os.path.join(SAVE_DIR, "svm_model.pkl"), "wb"))
np.save(os.path.join(SAVE_DIR, "img_shape.npy"), np.array(IMG_SIZE))

print("Saved to " + SAVE_DIR + "/")
print("  efficientnet_extractor.h5  svm_model.pkl  scaler.pkl  img_shape.npy")
print("Now run: python app.py")
