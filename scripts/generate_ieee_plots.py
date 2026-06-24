import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ==============================================================================
# IEEE STRICT FORMATTING INJECTION
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
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

os.makedirs("outputs/figures", exist_ok=True)

def parse_and_plot_high_res_loss():
    """Parses raw terminal telemetry to plot high-frequency batch loss vs continuous epochs."""
    log_file = "raw_loss_log.txt"
    
    if not os.path.exists(log_file):
        print(f"[ERROR] Cannot find '{log_file}'. Save your raw terminal output to this file.")
        return

    with open(log_file, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        # Regex to extract Epoch, Batch, Total Batches, and Loss
        match = re.search(r"Epoch (\d+)/\d+ \| Batch (\d+)/(\d+) \| Loss: ([\d\.]+)", line)
        if match:
            ep = int(match.group(1))
            batch = int(match.group(2))
            total_batches = int(match.group(3))
            loss = float(match.group(4))
            
            # Convert discrete batches into a continuous fractional epoch (e.g., Epoch 1 + 500/1088 = 1.45)
            # Subtract 1 from ep so Epoch 1 starts at 0.0
            continuous_epoch = (ep - 1) + (batch / total_batches)
            data.append((continuous_epoch, loss))

    if not data:
        print("[ERROR] Regex parser found no valid batch telemetry in the log file.")
        return

    df = pd.DataFrame(data, columns=["FractionalEpoch", "Loss"])
    
    # Calculate a rolling average to reveal the macroeconomic trend beneath the noise
    window_size = 5
    df['RollingAvg'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot the raw, noisy batch data (thinner, slightly transparent gray)
    ax.plot(df['FractionalEpoch'], df['Loss'], color='#666666', linewidth=1.5, alpha=0.7, 
            marker='x', markersize=6, label='Raw Batch Loss')
    
    # 2. Plot the smoothed rolling average over it (thick, solid black)
    ax.plot(df['FractionalEpoch'], df['RollingAvg'], color='black', linewidth=3.5, 
            label=f'Trend (MA-{window_size})')

    ax.set_title("TAT ARCHITECTURE: HIGH-RESOLUTION BATCH LOSS")
    ax.set_xlabel("EPOCH")
    ax.set_ylabel("LOSS")
    
    # Enforce strict integer ticks for the X-axis (Epoch 0, 1, 2, 3)
    max_epoch = int(np.ceil(df['FractionalEpoch'].max()))
    ax.set_xticks(range(0, max_epoch + 1))
    
    # Format X-axis labels to match human epoch counting (Epoch 1, 2, 3 instead of 0, 1, 2)
    ax.set_xticklabels([str(i+1) for i in range(0, max_epoch + 1)])
    
    ax.grid(True, linestyle='--', linewidth=1, alpha=0.5, color='gray')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(2)
    
    save_path = "outputs/figures/ieee_training_loss_high_res.png"
    plt.savefig(save_path)
    print(f"[PLOT] High-resolution loss curve saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parse_and_plot_high_res_loss()
    print("\n[SUCCESS] IEEE visual telemetry generated.")