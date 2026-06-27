import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
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
    if not os.path.exists(log_file): return

    with open(log_file, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        match = re.search(r"Epoch (\d+)/\d+ \| Batch (\d+)/(\d+) \| Loss: ([\d\.]+)", line)
        if match:
            ep, batch, total, loss = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
            data.append(((ep - 1) + (batch / total), loss))

    if not data: return
    df = pd.DataFrame(data, columns=["FractionalEpoch", "Loss"])
    df['RollingAvg'] = df['Loss'].rolling(window=5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vibrant Crimson for the raw data, Deep Navy for the trend
    ax.plot(df['FractionalEpoch'], df['Loss'], color='#FF4B4B', linewidth=1.5, alpha=0.5, label='Raw Batch Loss')
    ax.plot(df['FractionalEpoch'], df['RollingAvg'], color='#002244', linewidth=3.5, label='Trend (MA-5)')

    ax.set_title("TAT ARCHITECTURE: HIGH-RESOLUTION BATCH LOSS")
    ax.set_xlabel("EPOCH")
    ax.set_ylabel("LOSS")
    
    max_epoch = int(np.ceil(df['FractionalEpoch'].max()))
    ax.set_xticks(range(0, max_epoch + 1))
    ax.set_xticklabels([str(i+1) for i in range(0, max_epoch + 1)])
    
    ax.grid(True, linestyle='--', linewidth=1, alpha=0.5, color='gray')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(2)
    plt.savefig("outputs/figures/ieee_training_loss_high_res_color.png")
    plt.close()

def plot_comprehensive_survival_matrix():
    """Charts the full telemetry for MOT17-02, MOT17-04, and MOT17-09 in a 1x3 grid."""
    sequences = ["MOT17-02", "MOT17-04", "MOT17-09"]
    conditions = ["Clean", "Whitebox", "Blackbox"]
    
    baseline_data = {
        "MOT17-02": [99.6, 44.3, 56.5],
        "MOT17-04": [99.8, 2.2, 65.7],
        "MOT17-09": [90.5, 73.5, 89.0]
    }
    tat_data = {
        "MOT17-02": [99.6, 99.6, 99.6],
        "MOT17-04": [99.8, 99.8, 99.8],
        "MOT17-09": [95.8, 90.5, 90.5]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    x = np.arange(len(conditions))
    width = 0.35
    
    for i, seq in enumerate(sequences):
        ax = axes[i]
        base_rates = baseline_data[seq]
        tat_rates = tat_data[seq]
        
        rects1 = ax.bar(x - width/2, base_rates, width, label='Naive Baseline', 
                        color='#D32F2F', edgecolor='black', linewidth=2.5, hatch='//')
        rects2 = ax.bar(x + width/2, tat_rates, width, label='TAT Hardened', 
                        color='#1976D2', edgecolor='black', linewidth=2.5)
        
        ax.set_title(seq, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontweight='bold')
        ax.set_ylim(0, 115)
        
        if i == 0:
            ax.set_ylabel("SURVIVAL RATE (%)")
        
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=11, color='black')
            
    plt.tight_layout()
    fig.legend(['Naive Baseline', 'TAT Hardened'], loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, 
               edgecolor='black', fontsize=14, shadow=True).get_frame().set_linewidth(2)
    plt.subplots_adjust(top=0.82)
    
    save_path = "outputs/figures/ieee_survival_comprehensive.png"
    plt.savefig(save_path)
    print(f"[PLOT] Comprehensive survival matrix saved to {save_path}")
    plt.close()

def plot_trackeval_metrics():
    """Plots the official TrackEval MOTA and IDF1 scores proving macro-level stability."""
    sequences = ["MOT17-02", "MOT17-04", "MOT17-09"]
    
    # [Clean, Whitebox, Blackbox] extracted from TrackEval outputs
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
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(sequences))
    width = 0.25
    
    # IEEE Contrast Colors
    c_clean = '#4CAF50'     # Emerald Green
    c_whitebox = '#FF9800'  # Sunset Orange
    c_blackbox = '#9C27B0'  # Deep Purple
    
    # --- PLOT 1: MOTA ---
    ax1 = axes[0]
    ax1.bar(x - width, [mota_data[s][0] for s in sequences], width, color=c_clean, edgecolor='black', linewidth=2.5)
    ax1.bar(x, [mota_data[s][1] for s in sequences], width, color=c_whitebox, edgecolor='black', linewidth=2.5)
    ax1.bar(x + width, [mota_data[s][2] for s in sequences], width, color=c_blackbox, edgecolor='black', linewidth=2.5)
    
    ax1.set_title("MACRO-ROBUSTNESS: MOTA", pad=15)
    ax1.set_ylabel("MOTA (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sequences, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # --- PLOT 2: IDF1 ---
    ax2 = axes[1]
    ax2.bar(x - width, [idf1_data[s][0] for s in sequences], width, color=c_clean, edgecolor='black', linewidth=2.5)
    ax2.bar(x, [idf1_data[s][1] for s in sequences], width, color=c_whitebox, edgecolor='black', linewidth=2.5)
    ax2.bar(x + width, [idf1_data[s][2] for s in sequences], width, color=c_blackbox, edgecolor='black', linewidth=2.5)
    
    ax2.set_title("IDENTITY PRESERVATION: IDF1", pad=15)
    ax2.set_ylabel("IDF1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sequences, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add labels above the bars
    for ax, data_dict in zip([ax1, ax2], [mota_data, idf1_data]):
        for i, seq in enumerate(sequences):
            ax.annotate(f'{data_dict[seq][0]:.1f}', xy=(i - width, data_dict[seq][0]), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax.annotate(f'{data_dict[seq][1]:.1f}', xy=(i, data_dict[seq][1]), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax.annotate(f'{data_dict[seq][2]:.1f}', xy=(i + width, data_dict[seq][2]), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig.legend(['Clean Domain', 'Whitebox Attack', 'Blackbox Attack'], loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, 
               edgecolor='black', fontsize=14, shadow=True).get_frame().set_linewidth(2)
    plt.subplots_adjust(top=0.82)
    
    save_path = "outputs/figures/ieee_trackeval_metrics.png"
    plt.savefig(save_path)
    print(f"[PLOT] TrackEval MOTA/IDF1 macro-metrics saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print(f"[DEBUG] Checking output directory: {os.path.abspath('outputs/figures')}")
    plot_comprehensive_survival_matrix()
    print("[DEBUG] Matrix plot finished.")
    plot_comprehensive_survival_matrix()
    parse_and_plot_high_res_loss()
    plot_trackeval_metrics()
    print("\n[SUCCESS] All IEEE visual telemetry generated.")