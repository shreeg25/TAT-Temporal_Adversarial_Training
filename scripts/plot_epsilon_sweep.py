"""
Plot the epsilon sweep results (outputs/robustness_epsilon_sweep.csv) as a
single IEEE-style composite figure:

  - Top panel: macro-averaged (across sequences) target-ID survival % vs.
    epsilon, one line per condition (Clean / Whitebox / Blackbox). This is
    the headline robustness-vs-budget curve for the paper.
  - Bottom row: five small-multiple panels, one per MOT17 sequence, same
    three condition lines -- this is what shows the scene-dependent
    heterogeneity (MOT17-04/09 flat, MOT17-02 degrading, MOT17-10/13
    collapsed regardless of attack).

Run from the TAT-MNAT project root:
    python scripts/plot_epsilon_sweep.py

Reads:  outputs/robustness_epsilon_sweep.csv
Writes: outputs/figures/ieee_epsilon_sweep.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# IEEE STRICT FORMATTING INJECTION (matches generate_ieee_plots.py)
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
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CSV_PATH = "outputs/robustness_epsilon_sweep.csv"
OUT_PATH = "outputs/figures/ieee_epsilon_sweep.png"
TRAIN_EPSILON = 0.03137  # config.yaml adversarial_bounds.L_inf

CONDITION_STYLE = {
    "Clean":    {"color": "#1B7837", "marker": "o"},
    "Whitebox": {"color": "#B2182B", "marker": "s"},
    "Blackbox": {"color": "#2166AC", "marker": "^"},
}


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["seq_base"] = df["sequence"].str.replace("-FRCNN", "", regex=False)
    df["condition"] = "Clean"
    df.loc[df["seq_base"].str.endswith("-Whitebox"), "condition"] = "Whitebox"
    df.loc[df["seq_base"].str.endswith("-Blackbox"), "condition"] = "Blackbox"
    df["seq_base"] = df["seq_base"].str.replace("-Whitebox", "", regex=False)
    df["seq_base"] = df["seq_base"].str.replace("-Blackbox", "", regex=False)
    return df


def plot_condition_lines(ax, sub_df, show_legend=False):
    for condition, style in CONDITION_STYLE.items():
        line = sub_df[sub_df["condition"] == condition].sort_values("epsilon")
        if line.empty:
            continue
        ax.plot(
            line["epsilon"], line["survival_pct"],
            label=condition, color=style["color"], marker=style["marker"],
            markersize=5, linewidth=2.2,
        )
    ax.axvline(TRAIN_EPSILON, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(loc="lower left", fontsize=9, frameon=True)


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    df = load_data()
    sequences = sorted(df["seq_base"].unique())

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, len(sequences), height_ratios=[1.6, 1])

    # --- Top panel: macro-average across all sequences ---
    ax_macro = fig.add_subplot(gs[0, :])
    macro = df.groupby(["epsilon", "condition"], as_index=False)["survival_pct"].mean()
    plot_condition_lines(ax_macro, macro, show_legend=True)
    ax_macro.set_title("MACRO-AVERAGED TARGET-ID SURVIVAL vs. PERTURBATION BUDGET (ALL SEQUENCES)", pad=12)
    ax_macro.set_xlabel(r"Perturbation Budget ($\epsilon$, $L_\infty$)")
    ax_macro.set_ylabel("Target-ID Survival (%)")
    ax_macro.text(
        TRAIN_EPSILON, 102, "train budget", fontsize=8, color="gray",
        ha="center", fontweight="normal",
    )

    # --- Bottom row: one small-multiple panel per sequence ---
    for i, seq in enumerate(sequences):
        ax = fig.add_subplot(gs[1, i])
        plot_condition_lines(ax, df[df["seq_base"] == seq])
        ax.set_title(seq, fontsize=11, pad=8)
        ax.set_xlabel(r"$\epsilon$", fontsize=10)
        if i == 0:
            ax.set_ylabel("Survival (%)", fontsize=10)
        else:
            ax.set_yticklabels([])

    fig.suptitle("TAT-MNAT: ROBUSTNESS-BUDGET SWEEP", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    print(f"[SUCCESS] Figure saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
