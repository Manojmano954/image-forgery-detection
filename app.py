import os, io, pickle, numpy as np, cv2
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model

SAVE_DIR = "saved_model"
LABELS   = ["Original", "Tamper"]

app = Flask(__name__, static_folder="static")

print("Loading models...")
feature_extractor = load_model(os.path.join(SAVE_DIR, "efficientnet_extractor.h5"))
svm_model  = pickle.load(open(os.path.join(SAVE_DIR, "svm_model.pkl"), "rb"))
scaler     = pickle.load(open(os.path.join(SAVE_DIR, "scaler.pkl"), "rb"))
img_size   = tuple(np.load(os.path.join(SAVE_DIR, "img_shape.npy")).astype(int))
print(f"Models loaded. Image size: {img_size}")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Cannot decode image"}), 400

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img.astype("float32"), axis=0)

    features  = feature_extractor.predict(img_batch, verbose=0)
    features_scaled = scaler.transform(features)
    proba     = svm_model.predict_proba(features_scaled)[0]
    cls_idx   = int(np.argmax(proba))
    label     = LABELS[cls_idx]
    confidence = float(proba[cls_idx]) * 100

    return jsonify({
        "result":     label,
        "confidence": round(confidence, 2),
        "is_forged":  bool(cls_idx == 1),
        "probabilities": {
            "Original": round(float(proba[0]) * 100, 2),
            "Tamper":   round(float(proba[1]) * 100, 2),
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
