import os
import sys
import torch
import torchvision
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Force path recognition
sys.path.insert(0, os.path.abspath("."))

# ==============================================================================
# IEEE STRICT FORMATTING (matches your Batch A plots)
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
    "text.color": "black"
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hardened_model(weight_path):
    print(f"[*] Loading ResNet-50 FPN on {DEVICE}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model

# Global list to store the intercepted feature map
activation_maps = []

def forward_hook(module, input, output):
    """Intercepts the tensor output of the ResNet layer before it hits the FPN."""
    activation_maps.append(output)

def generate_heatmap(img_path, model, output_path):
    activation_maps.clear() # Clear previous hooks
    
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    
    # 1. Run inference (the hook will catch the activations)
    with torch.no_grad():
        _ = model(img_tensor)
        
    # 2. Extract the raw feature map from the hook
    # output shape is [1, 2048, H/32, W/32]
    feature_map = activation_maps[0].squeeze(0).cpu().numpy()
    
    # 3. Average across all 2048 channels to get the global activation energy
    heatmap = np.mean(feature_map, axis=0)
    
    # 4. Normalize the heatmap to 0-255
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    
    # 5. Resize to match the original image size
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    
    # 6. Apply Jet colormap and overlay
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap_color, 0.5, 0)
    
    # 7. Plot and Save using IEEE formatting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Plot: Raw Input
    ax[0].imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    ax[0].set_title("RAW INPUT (IDD UNSTRUCTURED FOLIAGE)", pad=15)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    # Right Plot: Heatmap
    ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax[1].set_title("RESNET-50 LAYER 4 (ADVERSARIAL SMOOTHING)", pad=15)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    fig.suptitle("INTERNAL FEATURE MAP COLLAPSE UNDER DOMAIN SHIFT", fontsize=18, y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved IEEE-formatted heatmap to {output_path}")

if __name__ == "__main__":
    WEIGHT_PATH = "weights/tat_hardened_mnat_epoch_10.pth"
    
    # TODO: Paste the exact path to one of the hallucinated tree images here.
    # For example, look at your output grid and find the path for 'det_Image_1001.png' or similar.
    TEST_IMAGE = "data/IDD_Detection/JPEGImages/frontFar/BLR-2018-04-16_16-14-27_frontFar/0006780.jpg" 
    
    OUTPUT_FILE = "outputs/figures/ieee_figure2_feature_map.png"
    os.makedirs("outputs/figures", exist_ok=True)
    
    model = load_hardened_model(WEIGHT_PATH)
    
    # Register the hook on the final bottleneck of ResNet-50
    model.backbone.body.layer4.register_forward_hook(forward_hook)
    
    print("[*] Generating Figure 2 (Feature Map)...")
    generate_heatmap(TEST_IMAGE, model, OUTPUT_FILE)