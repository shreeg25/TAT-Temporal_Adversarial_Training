"""
Quantitative feature-collapse metric.

generate_feature_maps.py gives you one qualitative heatmap on one IDD image.
This script turns that impression into numbers: for both the baseline
detector and the MNAT-hardened detector, on both MOT17 (in-domain) and IDD
(out-of-domain) images, it computes two metrics from the same layer4 hook:

  - spatial_variance : variance of the channel-averaged activation map
                       across spatial locations. Low = flat/diffuse
                       ("smoothed out", matching the qualitative heatmap).
  - channel_entropy   : Shannon entropy of the spatially-averaged,
                       normalized per-channel activation vector. Low =
                       activation energy concentrated in a small subset of
                       the 2048 channels ("collapsed" feature usage).

Four groups come out of this: {baseline, hardened} x {MOT17, IDD}. The
claim "feature collapse under domain shift, worse after adversarial
hardening" is supported if (hardened, IDD) has the lowest spatial_variance
and channel_entropy of the four -- and it's a real, checkable claim instead
of one image with a caption.

Run from the TAT-MNAT project root:
    python scripts/feature_collapse_metric.py

Reads:
    weights/faster_rcnn_mot17.pth              (baseline)
    weights/tat_hardened_mnat_epoch_10.pth     (MNAT-hardened)
    data/MOT17/train/MOT17-02-FRCNN/img1, MOT17-04-FRCNN/img1, MOT17-09-FRCNN/img1
    data/IDD_Detection/JPEGImages + val.txt

Writes:
    outputs/feature_collapse_metrics.csv
    outputs/figures/ieee_feature_collapse_metrics.png
"""

import os
import sys
import csv
import random
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, os.path.abspath("."))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IMAGES_PER_DOMAIN = 40  # sampled per domain, per model
RANDOM_SEED = 42

MODELS = {
    "Baseline":  "weights/faster_rcnn_mot17.pth",
    "MNAT-Hardened": "weights/tat_hardened_mnat_epoch_10.pth",
}

MOT17_SEQS = [
    "data/MOT17/train/MOT17-02-FRCNN/img1",
    "data/MOT17/train/MOT17-04-FRCNN/img1",
    "data/MOT17/train/MOT17-09-FRCNN/img1",
]
IDD_BASE = "data/IDD_Detection"
IDD_IMG_DIR = os.path.join(IDD_BASE, "JPEGImages")
IDD_SPLIT_FILE = os.path.join(IDD_BASE, "val.txt")


def load_model(weight_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model


def sample_mot17_images(n):
    all_paths = []
    for img_dir in MOT17_SEQS:
        if not os.path.isdir(img_dir):
            print(f"  [SKIP] Missing MOT17 dir: {img_dir}")
            continue
        all_paths += [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith(".jpg")]
    random.Random(RANDOM_SEED).shuffle(all_paths)
    return all_paths[:n]


def sample_idd_images(n):
    if not os.path.exists(IDD_SPLIT_FILE):
        print(f"  [SKIP] Missing IDD split file: {IDD_SPLIT_FILE}")
        return []
    with open(IDD_SPLIT_FILE) as f:
        names = [line.strip() for line in f if line.strip()]
    random.Random(RANDOM_SEED).shuffle(names)
    paths = []
    for name in names:
        p = os.path.join(IDD_IMG_DIR, f"{name}.jpg")
        if os.path.exists(p):
            paths.append(p)
        if len(paths) >= n:
            break
    return paths


def compute_metrics(model, img_path, activation_store):
    activation_store.clear()
    img = Image.open(img_path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = model(tensor)

    feat = activation_store[0].squeeze(0).cpu().numpy()  # [C, H, W]
    feat = np.maximum(feat, 0)  # post-ReLU should already be >=0; guard anyway

    spatial_map = feat.mean(axis=0)  # [H, W]
    spatial_variance = float(np.var(spatial_map))

    channel_vec = feat.mean(axis=(1, 2))  # [C]
    total = channel_vec.sum()
    if total <= 0:
        channel_entropy = 0.0
    else:
        p = channel_vec / total
        channel_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return spatial_variance, channel_entropy


def main():
    random.seed(RANDOM_SEED)
    os.makedirs("outputs/figures", exist_ok=True)

    mot17_paths = sample_mot17_images(N_IMAGES_PER_DOMAIN)
    idd_paths = sample_idd_images(N_IMAGES_PER_DOMAIN)
    print(f"[INFO] Sampled {len(mot17_paths)} MOT17 images, {len(idd_paths)} IDD images.")

    rows = []
    for model_name, weight_path in MODELS.items():
        if not os.path.exists(weight_path):
            print(f"[SKIP] Missing weights: {weight_path}")
            continue
        print(f"\n[MODEL] {model_name}")
        model = load_model(weight_path)
        activation_store = []
        model.backbone.body.layer4.register_forward_hook(lambda m, i, o: activation_store.append(o))

        for domain, paths in [("MOT17", mot17_paths), ("IDD", idd_paths)]:
            for img_path in paths:
                try:
                    sv, ce = compute_metrics(model, img_path, activation_store)
                except Exception as e:
                    print(f"  [ERROR] {img_path}: {e}")
                    continue
                rows.append({
                    "model": model_name, "domain": domain, "image": img_path,
                    "spatial_variance": sv, "channel_entropy": ce,
                })
            print(f"  [{domain}] done ({len(paths)} images)")

    csv_path = "outputs/feature_collapse_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "domain", "image", "spatial_variance", "channel_entropy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SUCCESS] Raw metrics saved to {csv_path}")

    plot_summary(rows)


def plot_summary(rows):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.edgecolor": "black", "axes.linewidth": 2.5,
        "xtick.major.width": 2, "ytick.major.width": 2,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.labelsize": 14, "axes.titlesize": 14,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Preserve a stable, readable group order regardless of dict insertion order
    ordered_keys = [
        ("Baseline", "MOT17"), ("Baseline", "IDD"),
        ("MNAT-Hardened", "MOT17"), ("MNAT-Hardened", "IDD"),
    ]
    groups = {}
    for r in rows:
        key = (r["model"], r["domain"])
        groups.setdefault(key, {"sv": [], "ce": []})
        groups[key]["sv"].append(r["spatial_variance"])
        groups[key]["ce"].append(r["channel_entropy"])
    ordered_keys = [k for k in ordered_keys if k in groups]

    # Short two-line labels avoid the horizontal collision that comes from
    # cramming "MNAT-Hardened\nIDD" into the same width as "Baseline\nMOT17"
    label_map = {"Baseline": "Baseline", "MNAT-Hardened": "MNAT-\nHardened"}
    labels = [f"{label_map[m]}\n{d}" for (m, d) in ordered_keys]
    sv_means = [np.mean(groups[k]["sv"]) for k in ordered_keys]
    sv_stds = [np.std(groups[k]["sv"]) for k in ordered_keys]
    ce_means = [np.mean(groups[k]["ce"]) for k in ordered_keys]
    ce_stds = [np.std(groups[k]["ce"]) for k in ordered_keys]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(labels))
    bar_width = 0.6

    axes[0].bar(x, sv_means, yerr=sv_stds, capsize=5, color="#B2182B", width=bar_width)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10, linespacing=1.4)
    axes[0].set_xlim(-0.6, len(labels) - 0.4)
    axes[0].set_title("SPATIAL ACTIVATION VARIANCE (LAYER 4)")
    axes[0].set_ylabel("Variance (lower = flatter/collapsed)")

    axes[1].bar(x, ce_means, yerr=ce_stds, capsize=5, color="#2166AC", width=bar_width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10, linespacing=1.4)
    axes[1].set_xlim(-0.6, len(labels) - 0.4)
    axes[1].set_title("CHANNEL ENTROPY (LAYER 4)")
    axes[1].set_ylabel("Entropy (lower = fewer channels active)")

    fig.suptitle("FEATURE COLLAPSE: BASELINE vs. MNAT-HARDENED, IN-DOMAIN vs. OUT-OF-DOMAIN", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig("outputs/figures/ieee_feature_collapse_metrics.png")
    print("[SUCCESS] Figure saved to outputs/figures/ieee_feature_collapse_metrics.png")


if __name__ == "__main__":
    main()
