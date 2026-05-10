"""
visualize_preprocessing.py
==========================
Run all preprocessing techniques on one sample image per class and save grids.

Usage:
    python src/visualize_preprocessing.py --data_dir dataset/Images
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import visualize_preprocessing


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    train_root = os.path.join(args.data_dir, "Train")
    for cls in sorted(os.listdir(train_root)):
        cls_dir = os.path.join(train_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        # pick first valid image
        img_path = None
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(cls_dir, f)
                break
        if img_path is None:
            continue

        out_path = os.path.join(args.out_dir, f"preprocessing_{cls}.png")
        print(f"[visualize] Processing class '{cls}' — {img_path}")
        visualize_preprocessing(img_path, out_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/Images")
    parser.add_argument("--out_dir",  default="outputs")
    args = parser.parse_args()
    main(args)
