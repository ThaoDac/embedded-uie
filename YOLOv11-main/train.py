#!pip install ultralytics==8.3.49
# ============================================================
# urpc/
#  ├── images/
#  │    ├── train/
#  │    ├── val/
#  ├── labels/
#  │    ├── train/
#  │    ├── val/
#  └── urpc.yaml
# ============================================================

from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Load pretrained YOLOv11-nano
# ============================================================
model = YOLO("/home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/yolo11n.pt")   # dùng pretrained để fine-tune

# ============================================================
# 2. Fine-tune model
# ============================================================
results = model.train(
    data="/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml",
    epochs=150,
    imgsz=640,
    batch=1,
    device=0,               # GPU = 0, CPU = 'cpu'
    optimizer="AdamW",
    lr0=0.001,
    patience=30,            # early stopping
    workers=4,
    project="urpc_train",
    name="yolov11n_finetune",
    verbose=True
)

# ============================================================
# 3. Lấy toàn bộ logs của YOLO (loss + metrics)
# ============================================================
metrics_file = "urpc_train/yolov11n_finetune/results.csv"

df = pd.read_csv(metrics_file)

print("\n===== METRICS CÓ SẴN TRONG YOLO =====")
print(df.columns)


# ============================================================
# 4. Tính F1-score theo Precision & Recall
# ============================================================
df["F1"] = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"] + 1e-8)


# ============================================================
# 5. In một số thông số quan trọng
# ============================================================
print("\n===== KẾT QUẢ CUỐI CÙNG =====")
print("Train Loss:", df["train/box_loss"].iloc[-1])
print("Val Loss:", df["val/box_loss"].iloc[-1])
print("Precision:", df["precision"].iloc[-1])
print("Recall:", df["recall"].iloc[-1])
print("F1-score:", df["F1"].iloc[-1])
print("mAP@50:", df["map50"].iloc[-1])
print("mAP@50-95:", df["map"].iloc[-1])


# ============================================================
# 6. VẼ ĐỒ THỊ LOSS + METRICS
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
plt.legend(); plt.title("LOSS QUÁ TRÌNH TRAIN"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(); plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["precision"], label="Precision")
plt.plot(df["epoch"], df["recall"], label="Recall")
plt.plot(df["epoch"], df["F1"], label="F1-score")
plt.legend(); plt.title("PRECISION / RECALL / F1"); plt.xlabel("Epoch")
plt.grid(); plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["map50"], label="mAP50")
plt.plot(df["epoch"], df["map"], label="mAP50-95")
plt.legend(); plt.title("mAP"); plt.xlabel("Epoch")
plt.grid(); plt.show()
