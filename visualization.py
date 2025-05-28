import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("runs/train/blueprint_yolo_local/results.csv")

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('YOLOv8 Training Metrics')

# 1. Training and Validation Losses
axs[0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
axs[0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
axs[0].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
axs[0].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
axs[0].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
axs[0].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
axs[0].set_title("Loss Curves")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

# 2. Precision & Recall
axs[1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
axs[1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
axs[1].set_title("Precision & Recall")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Score")
axs[1].legend()
axs[1].grid(True)

# 3. mAP Curves
axs[2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
axs[2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
axs[2].set_title("mAP Metrics")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Score")
axs[2].legend()
axs[2].grid(True)

# Save and show
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("training_metrics.png")
plt.show()
