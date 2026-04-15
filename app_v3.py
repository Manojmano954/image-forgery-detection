import os, cv2, numpy as np, base64, torch, torch.nn as nn, torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms as T

SAVE_DIR     = "saved_model"
MODEL_PATH   = os.path.join(SAVE_DIR, "ManTraNet.pt")
IMG_SIZE     = 64
app          = Flask(__name__, static_folder="static")
model        = None
model_loaded = False
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── EXACT architecture used during training ────────────────────────────────────

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

        # Decoder
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
        noise   = torch.cat([x, self.bayar(x), self.srm(x)], dim=1)
        e1      = self.enc1(noise)
        e2      = self.enc2(self.pool1(e1))
        e3      = self.enc3(self.pool2(e2))
        bn      = self.bottleneck(self.pool3(e3))
        cls_out = self.cls_head(self.gap(bn))
        d3      = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2      = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1      = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_out = self.seg_head(d1)          # raw logits (BCEWithLogitsLoss trained)
        return cls_out, seg_out


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    global model, model_loaded
    m = ManTraNetV3().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            m.load_state_dict(state)
            print("✓ ManTraNet.pt loaded from", MODEL_PATH)
        except Exception as e:
            print("✗ Could not load weights:", e)
    else:
        print("✗ ManTraNet.pt not found in", SAVE_DIR)
        print("  Download it from Google Drive and place in saved_model/")
    m.eval()
    model = m
    model_loaded = True
    print("ManTraNet V3 ready on", DEVICE)

load_model()


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(img_rgb):
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil = Image.fromarray(img_rgb).convert("RGB")
    inp = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_out, seg_out = model(inp)

    # Classification result
    probs      = torch.softmax(cls_out, dim=1).squeeze().cpu().numpy()
    pred_class = int(cls_out.argmax(1).item())   # 0=Original, 1=Tamper
    is_forged  = pred_class == 1
    label      = "Tamper" if is_forged else "Original"
    confidence = round(float(probs[pred_class]) * 100, 2)

    # Segmentation heatmap (sigmoid on logits)
    heatmap = torch.sigmoid(seg_out).squeeze().cpu().numpy()
    lo, hi  = heatmap.min(), heatmap.max()
    heatmap = (heatmap - lo) / (hi - lo + 1e-8)

    return label, confidence, heatmap, is_forged, probs


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def overlay_heatmap(heatmap, img_rgb):
    h, w  = img_rgb.shape[:2]
    hm    = cv2.resize(heatmap, (w, h))
    hm_c  = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over  = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.55, hm_c, 0.45, 0)
    _, b  = cv2.imencode(".png", over)
    return base64.b64encode(b).decode("utf-8")


def highlight_regions(heatmap, img_rgb, threshold=0.6):
    h, w    = img_rgb.shape[:2]
    hm      = cv2.resize(heatmap, (w, h))
    mask    = (hm > threshold).astype(np.uint8) * 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 3)
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x+rw, y+rh), (0, 255, 255), 2)
        cv2.putText(result, "Forged Region", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    _, b = cv2.imencode(".png", result)
    return base64.b64encode(b).decode("utf-8")


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    base = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base, "index.html")

@app.route("/model_status")
def status():
    return jsonify({
        "loaded":  model_loaded,
        "model":   "ManTraNet V3",
        "device":  DEVICE,
        "weights": os.path.exists(MODEL_PATH)
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    fb      = np.frombuffer(request.files["image"].read(), np.uint8)
    img_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image"}), 400
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        label, conf, heatmap, is_forged, probs = run_inference(img_rgb)
        hm_b64  = overlay_heatmap(heatmap, img_rgb)
        box_b64 = highlight_regions(heatmap, img_rgb) if is_forged else None
        return jsonify({
            "result":     label,
            "confidence": conf,
            "is_forged":  is_forged,
            "heatmap":    hm_b64,
            "regions":    box_b64,
            "probabilities": {
                "Original": round(float(probs[0]) * 100, 2),
                "Tamper":   round(float(probs[1]) * 100, 2),
            },
            "model":         "ManTraNet V3",
            "forgery_types": 385
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("─" * 45)
    print("  ForgeGuard V3 — http://localhost:5000")
    print("─" * 45)
    app.run(debug=True, host="0.0.0.0", port=5000)
