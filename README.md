# image-forgery-detection



\# Image Forgery Detection System

\*\*MSc Data Science — South East Technological University\*\*



\## Team

| Member | Role | Files |

|--------|------|-------|

| Manoj Kumar Konda | Model Training | train.py, train\_v3.py |

| Ranganayakulu | Web Application | app.py, index.html |



\## Version History



\### V1 — MobileNetV2 (branch: v1-mobilenetv2)

\- Model: MobileNetV2 + RBF-SVM

\- Dataset: MICC-F220 (220 images)

\- links: 

\- Detects: Copy-Move forgery only

\- Accuracy: \~77%



\### V2 — EfficientNetB0 (branch: v2-efficientnet)

\- Model: EfficientNetB0 + RBF-SVM

\- Dataset: CASIA v2 + CoMoFoD + Columbia (24,459 images)

\- Detects: Copy-Move + Splicing

\- Accuracy: 81.34%



\### V3 — ManTraNet Style (branch: v3-mantranet)

\- Model: BayarConv + SRM + U-Net segmentation

\- Dataset: CASIA v2 + CoMoFoD + Columbia + FaceForensics++(download deepfake\_dataset.py and you will get the dataset)python deepfake\_dataset.py FaceForensics -d Deepfakes -c c23 -t videos -n 50 --server EU2

\- Detects: Copy-Move + Splicing + Deepfakes

\- Output: Classification + Pixel heatmap showing WHERE forgery is



\## How to Run

pip install -r requirements.txt

python train.py

python app.py

Open http://localhost:5000



