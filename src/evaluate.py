"""
evaluate.py
===========
Evaluate a trained checkpoint on the test set and print a classification report.

Usage:
    python src/evaluate.py --checkpoint outputs/best_model.pth --data_dir dataset/Images
"""

import argparse
import json
import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders, get_val_transforms, CarDataset
from model   import build_model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = model(imgs)

        """ out = tensor([
      [ 2.5, -1.2,  0.1],  # Image 1: Highest score is 2.5 (Index 0: Sedan)
      [-0.5,  4.8,  1.2],  # Image 2: Highest score is 4.8 (Index 1: SUV)
      [ 0.2,  0.5,  3.9]   # Image 3: Highest score is 3.9 (Index 2: Coupe)])"""
        
        preds = out.argmax(dim=1).cpu().numpy()
        #out.argmax(dim=1) (The Index): This finds that the largest number (\(2.5\)) is at position 0. So, preds becomes 0.
        #You need .cpu() because NumPy cannot handle tensors that are stored on a GPU
        #Why numpy? → sklearn's functions expect numpy arrays, not PyTorch tensors

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

        """"
        PyTorch (GPU): Does the fast, complex prediction math.
        NumPy (CPU): Formats those predictions into a clean, simple list.
        Sklearn (CPU): Analyzes that list to give you a professional performance grade.
        """
        """
        A GPU is designed for Parallelism. 
        Instead of one fast accountant, a GPU is like having 5,000 tiny calculators working at the same time:
        Simultaneous Math: The GPU can calculate the top-left corner of the image and the bottom-right corner at the exact same millisecond.
        Throughput: This makes evaluation (inference) significantly faster. On a CPU, one image might take 200ms; on a GPU, it might take 5ms.
        Batching: This is why you have a batch_size (defaulting to 16 in your code).
          The GPU processes all 16 images at once in a single mathematical "sweep."
        """
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix – Car Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx: dict = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    #rchitecture (build_model) + Weights (load_state_dict) = complete trained model
    print(f"[evaluate] Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_acc={ckpt['val_acc']*100:.1f}%)")

    # Data
    test_root = os.path.join(args.data_dir, "Test")
    test_ds   = CarDataset(test_root, get_val_transforms(args.img_size), class_to_idx)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Predict
    preds, labels = predict_all(model, test_loader, device)
    acc = (preds == labels).mean() * 100

    print(f"\n[evaluate] Test Accuracy: {acc:.1f}%\n")
    print(classification_report(labels, preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_confusion_matrix(cm, class_names,
                          os.path.join(args.out_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--data_dir",   default="dataset/Images")
    parser.add_argument("--out_dir",    default="outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size",   type=int, default=224)
    args = parser.parse_args()
    main(args)
