"""
Batch A plots for the TAT-MNAT ICVGIP submission.

All four plots here are built from data that already exists on disk (logs,
rendered frames, saved images) — nothing is fabricated. Where a required
input file is missing, the function prints exactly what to run to produce
it and skips that plot instead of guessing numbers.

Run from the project root:
    python scripts/generate_ieee_plots_batchA.py
"""
import os
import re
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ==============================================================================
# IEEE STRICT FORMATTING (matches scripts/generate_ieee_plots.py)
# ==============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 2.5,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

C_CLEAN = '#4CAF50'
C_WHITEBOX = '#FF9800'
C_BLACKBOX = '#9C27B0'


# ==============================================================================
# PLOT 1: HOTA / MOTA / IDF1 grouped bars
# ==============================================================================
def _find_trackeval_summary(tracker_root, seq_name=None):
    """
    Looks for TrackEval's own output: <tracker_root>/[seq_name/]pedestrian_summary.txt
    (space-delimited: header row of field names, then one row of values).
    Returns a dict of {field: float} or None if not found.
    """
    if seq_name:
        path = os.path.join(tracker_root, seq_name, "pedestrian_summary.txt")
    else:
        path = os.path.join(tracker_root, "pedestrian_summary.txt")

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 2:
        return None

    fields = lines[0].split()
    values = lines[1].split()
    out = {}
    for k, v in zip(fields, values):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def plot_hota_mota_idf1():
    """
    Grouped bars of HOTA / MOTA / IDF1 across sequences x {Clean, Whitebox, Blackbox}.

    First choice: read real numbers from TrackEval output
        TrackEval/data/trackers/mot_challenge/MOT17-train/<tracker>/<seq>/pedestrian_summary.txt
    This requires TrackEval's eval.py to have actually been run against exported
    tracker results for both the baseline and TAT-hardened trackers, on each
    condition (Clean / Whitebox / Blackbox), per sequence.

    Fallback: reuse the known-good MOTA/IDF1 numbers already hardcoded in
    scripts/generate_ieee_plots.py (these came from a real prior TrackEval run)
    and skip the HOTA row with a printed warning, rather than inventing a number.
    """
    sequences = ["MOT17-02", "MOT17-04", "MOT17-09"]
    conditions = ["Clean", "Whitebox", "Blackbox"]
    suffix_map = {"Clean": "-FRCNN", "Whitebox": "-FRCNN-Whitebox", "Blackbox": "-FRCNN-Blackbox"}

    tracker_root = "TrackEval/data/trackers/mot_challenge/MOT17-train/TAT_MNAT_Architecture"

    hota_data, mota_data, idf1_data = {}, {}, {}
    found_real_hota = True

    for seq in sequences:
        hota_data[seq], mota_data[seq], idf1_data[seq] = [], [], []
        for cond in conditions:
            seq_name = seq + suffix_map[cond]
            summary = _find_trackeval_summary(tracker_root, seq_name)
            if summary and "HOTA" in summary:
                hota_data[seq].append(summary["HOTA"])
                mota_data[seq].append(summary.get("MOTA", np.nan))
                idf1_data[seq].append(summary.get("IDF1", np.nan))
            else:
                found_real_hota = False
                hota_data[seq].append(np.nan)
                mota_data[seq].append(np.nan)
                idf1_data[seq].append(np.nan)

    if not found_real_hota:
        print("[WARN] No real TrackEval pedestrian_summary.txt files found under "
              f"{tracker_root}/<seq>/. Falling back to the known MOTA/IDF1 numbers "
              "from scripts/generate_ieee_plots.py and omitting the HOTA panel.")
        print("        To get real HOTA: run scripts/export_trackeval.py for each "
              "condition, then TrackEval/scripts/run_mot_challenge.py, which writes "
              "pedestrian_summary.txt per sequence.")
        mota_data = {
            "MOT17-02": [45.97, 46.19, 45.44],
            "MOT17-04": [75.38, 75.40, 75.40],
            "MOT17-09": [65.07, 64.81, 65.20]
        }
        idf1_data = {
            "MOT17-02": [44.29, 43.43, 43.75],
            "MOT17-04": [77.11, 77.09, 77.10],
            "MOT17-09": [48.45, 47.23, 45.24]
        }
        hota_data = None  # signal: don't plot this panel

    x = np.arange(len(sequences))
    width = 0.25
    n_panels = 3 if hota_data is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 2:
        axes = list(axes)

    panels = []
    if hota_data is not None:
        panels.append(("HOTA", "HOTA (%)", hota_data))
    panels.append(("MACRO-ROBUSTNESS: MOTA", "MOTA (%)", mota_data))
    panels.append(("IDENTITY PRESERVATION: IDF1", "IDF1 (%)", idf1_data))

    for ax, (title, ylabel, data_dict) in zip(axes, panels):
        ax.bar(x - width, [data_dict[s][0] for s in sequences], width,
               color=C_CLEAN, edgecolor='black', linewidth=2.5)
        ax.bar(x, [data_dict[s][1] for s in sequences], width,
               color=C_WHITEBOX, edgecolor='black', linewidth=2.5)
        ax.bar(x + width, [data_dict[s][2] for s in sequences], width,
               color=C_BLACKBOX, edgecolor='black', linewidth=2.5)
        ax.set_title(title, pad=15)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, fontweight='bold')
        ax.set_ylim(0, 100)
        for i, seq in enumerate(sequences):
            for off, val in zip([-width, 0, width], data_dict[seq]):
                if np.isnan(val):
                    continue
                ax.annotate(f'{val:.1f}', xy=(i + off, val), xytext=(0, 5),
                            textcoords="offset points", ha='center', va='bottom',
                            fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig.legend(['Clean Domain', 'Whitebox Attack', 'Blackbox Attack'], loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=True,
               edgecolor='black', fontsize=13, shadow=True).get_frame().set_linewidth(2)
    plt.subplots_adjust(top=0.80)

    save_path = os.path.join(FIG_DIR, "ieee_hota_mota_idf1.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


# ==============================================================================
# PLOT 2: Full 5-sequence survival heatmap (Baseline + TAT, all conditions)
# ==============================================================================
def _parse_survival_matrix_txt(path):
    """Parses the '<seq> | <id> | survived/total | rate%' table format."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"^(MOT17-\d+-FRCNN(?:-Whitebox|-Blackbox)?)\s*\|.*\|\s*([\d.]+)%", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def _parse_robustness_log_txt(path):
    """Parses 'SEQ-NAME: survived/total (rate%)' format."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"^(MOT17-\d+-FRCNN(?:-Whitebox|-Blackbox)?):\s*\d+/\d+\s*\(([\d.]+)%\)", line.strip())
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def plot_full_survival_heatmap():
    """
    Heatmap of target-ID survival rate across ALL evaluated sequences
    (02/04/09/10/13) x {Baseline, TAT-Hardened} x {Clean, Whitebox, Blackbox}.

    Baseline+TAT comparison for 02/04/09 comes from
        outputs/survival_matrix_faster_rcnn_mot17_baseline.txt
        outputs/survival_matrix_faster_rcnn_mot17.txt
    TAT-only numbers for 10/13 come from
        outputs/robustness_eps_0_1.txt
    (no baseline run exists yet for 10/13 in this repo — those cells are left
    blank/NaN rather than invented).
    """
    baseline = _parse_survival_matrix_txt("outputs/survival_matrix_faster_rcnn_mot17_baseline.txt")
    tat = _parse_survival_matrix_txt("outputs/survival_matrix_faster_rcnn_mot17.txt")
    robustness = _parse_robustness_log_txt("outputs/robustness_eps_0_1.txt")

    # Merge: prefer the dedicated survival-matrix files; fill gaps from the
    # robustness log (which covers 10/13 but only for the TAT model).
    tat_full = {**robustness, **tat}

    seq_bases = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT17-10", "MOT17-13"]
    conditions = [("Clean", ""), ("Whitebox", "-Whitebox"), ("Blackbox", "-Blackbox")]

    row_labels = []
    rows = []
    for model_name, source in [("Baseline", baseline), ("TAT-Hardened", tat_full)]:
        for cond_name, cond_suffix in conditions:
            row_labels.append(f"{model_name} / {cond_name}")
            row = []
            for seq in seq_bases:
                key = f"{seq}-FRCNN{cond_suffix}"
                row.append(source.get(key, np.nan))
            rows.append(row)

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='#DDDDDD')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(np.arange(len(seq_bases)))
    ax.set_xticklabels(seq_bases, fontweight='bold')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{val:.1f}%" if not np.isnan(val) else "N/A"
            color = "black" if not np.isnan(val) and val > 50 else ("white" if not np.isnan(val) else "gray")
            ax.text(j, i, text, ha="center", va="center", fontweight='bold',
                    fontsize=10, color=color)

    ax.axhline(2.5, color='black', linewidth=3)  # separates Baseline block from TAT block
    ax.set_title("TARGET-ID SURVIVAL RATE — ALL EVALUATED SEQUENCES", pad=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Survival Rate (%)")

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "ieee_full_survival_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved {save_path}")
    print("[NOTE] MOT17-10 and MOT17-13 baseline cells are blank (N/A) — no baseline "
          "run exists for those in this repo yet. Both sequences also show very low "
          "TAT survival (MOT17-13: 0%), which is worth a dedicated failure-case "
          "discussion (likely crowd density / occlusion) in the paper.")


# ==============================================================================
# PLOT 3: Qualitative tracking grid from rendered frames
# ==============================================================================
def plot_qualitative_tracking_grid(frames_per_seq=4):
    """
    Builds a montage of sampled frames (with drawn tracking boxes) for each
    sequence found under outputs/render_frames/. Only Whitebox-condition
    frames currently exist in this repo (render_telemetry.py was only run for
    the whitebox suite) — this plots what's available and prints how to
    generate Clean/Blackbox frames for a true side-by-side comparison.
    """
    render_root = "outputs/render_frames"
    seq_dirs = sorted(glob.glob(os.path.join(render_root, "*")))
    seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]

    if not seq_dirs:
        print(f"[SKIP] No rendered frames found under {render_root}/. "
              "Run scripts/render_telemetry.py first.")
        return

    print("[NOTE] Only Whitebox-condition renders exist in this repo. For a true "
          "Clean vs Whitebox vs Blackbox qualitative comparison, point "
          "render_telemetry.py's target_seqs at the -FRCNN and -FRCNN-Blackbox "
          "sequences too, then re-run this plot.")

    n_seqs = len(seq_dirs)
    fig, axes = plt.subplots(n_seqs, frames_per_seq, figsize=(4 * frames_per_seq, 4 * n_seqs))
    if n_seqs == 1:
        axes = np.array([axes])

    for row, seq_dir in enumerate(seq_dirs):
        seq_name = os.path.basename(seq_dir)
        frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
        if not frame_files:
            continue
        sample_idx = np.linspace(0, len(frame_files) - 1, frames_per_seq).astype(int)
        for col, idx in enumerate(sample_idx):
            ax = axes[row, col]
            img = Image.open(frame_files[idx])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(seq_name, fontsize=12, fontweight='bold')
            frame_no = os.path.splitext(os.path.basename(frame_files[idx]))[0]
            ax.set_title(f"frame {frame_no}", fontsize=10)

    fig.suptitle("QUALITATIVE TRACKING UNDER WHITEBOX ATTACK (TAT-Hardened)", fontsize=16, y=1.0)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "ieee_qualitative_tracking_grid.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


# ==============================================================================
# PLOT 4: IDD zero-shot generalization (qualitative grid + precision/recall)
# ==============================================================================
def plot_idd_zero_shot_qualitative(n_images=12):
    img_dir = "outputs/IDD_zero_shot"
    imgs = sorted(glob.glob(os.path.join(img_dir, "det_*.png")))
    if not imgs:
        print(f"[SKIP] No IDD zero-shot detection images found under {img_dir}/. "
              "Run scripts/idd_evaluate_zero_shot.py first.")
        return

    sample = imgs[:n_images] if len(imgs) >= n_images else imgs
    n_cols = 4
    n_rows = int(np.ceil(len(sample) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.axis('off')
        if i < len(sample):
            ax.imshow(Image.open(sample[i]))
            ax.set_title(os.path.basename(sample[i]), fontsize=8)

    fig.suptitle("ZERO-SHOT DOMAIN GENERALIZATION: TAT-HARDENED DETECTOR ON IDD", fontsize=15, y=1.0)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "ieee_idd_zero_shot_qualitative.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_idd_zero_shot_precision_recall():
    metrics_path = "outputs/idd_zero_shot_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"[SKIP] {metrics_path} not found. Run the (now-patched) "
              "scripts/idd_evaluate_zero_shot.py first — it now saves this JSON "
              "automatically at the end of a run.")
        return

    with open(metrics_path, "r") as f:
        m = json.load(f)

    labels = ["Precision", "Recall", "F1"]
    values = [m["precision"] * 100, m["recall"] * 100, m["f1"] * 100]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, values, color=[C_CLEAN, C_WHITEBOX, C_BLACKBOX],
                   edgecolor='black', linewidth=2.5, width=0.5)
    for rect, val in zip(bars, values):
        ax.annotate(f'{val:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 6), textcoords="offset points", ha='center',
                    va='bottom', fontweight='bold', fontsize=13)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("ZERO-SHOT IDD DETECTION METRICS", pad=15)
    ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.5, color='gray')

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "ieee_idd_zero_shot_precision_recall.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


if __name__ == "__main__":
    print(f"[DEBUG] Output directory: {os.path.abspath(FIG_DIR)}")
    plot_hota_mota_idf1()
    plot_full_survival_heatmap()
    plot_qualitative_tracking_grid()
    plot_idd_zero_shot_qualitative()
    plot_idd_zero_shot_precision_recall()
    print("\n[SUCCESS] Batch A plotting pass complete.")
