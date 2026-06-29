import sys
import os
import yaml
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, os.path.abspath("."))
from src.device import DEVICE

TARGET_MAP = {"MOT17-02": 19, "MOT17-04": 1, "MOT17-09": 12}

# --- RESTORED: Spatial IoU Calculator ---
def get_iou(bb1, bb2):
    x_left, y_top = max(bb1[0], bb2[0]), max(bb1[1], bb2[1])
    x_right, y_bottom = min(bb1[0] + bb1[2], bb2[0] + bb2[2]), min(bb1[1] + bb1[3], bb2[1] + bb2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    return intersection / float(bb1[2] * bb1[3] + bb2[2] * bb2[3] - intersection + 1e-8)

def load_hardened_detector(weight_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    return model

def evaluate_sequence(seq_path, model, target_id, epsilon=0.1):
    img_dir = os.path.join(seq_path, "img1")
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    df = pd.read_csv(gt_file, header=None, names=["frame", "id", "x", "y", "w", "h", "active", "class", "vis"])
    target_df = df[(df["id"] == target_id) & (df["class"] == 1)]
    gt_trajectory = {int(row["frame"]): [row["x"], row["y"], row["w"], row["h"]] for _, row in target_df.iterrows()}
    
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=(DEVICE.type=='cuda'))
    
    survived_frames = 0
    for frame_name in frames:
        frame_no = int(frame_name.split('.')[0])
        if frame_no not in gt_trajectory: continue
        
        img_path = os.path.join(img_dir, frame_name)
        
        # --- RESTORED: Image Loading and Tensor Definition ---
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).float().div_(255.0).to(DEVICE).unsqueeze(0)
        
        # --- PGD Attack Injection (Safe Mode) ---
        model.eval() 
        tensor.requires_grad = True
        
        output = model(tensor)
        loss = output[0]['scores'].sum() 
        
        model.zero_grad()
        loss.backward()
        
        perturbed_tensor = (tensor + epsilon * tensor.grad.sign()).detach().clamp(0, 1)
        
        # --- Tracking Inference ---
        with torch.no_grad():
            final_preds = model(perturbed_tensor)[0]
        
        keep = (final_preds["labels"] == 1) & (final_preds["scores"] > 0.4)
        boxes = final_preds["boxes"][keep].cpu().numpy()
        scores = final_preds["scores"][keep].cpu().numpy()
        
        xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
        
        # --- RESTORED: Passing rgb to frame argument ---
        tracks = tracker.update_tracks([[d, float(s), "1"] for d, s in zip(xywh, scores)], frame=rgb)
        
        if any(get_iou(t.to_tlwh(), gt_trajectory[frame_no]) >= 0.45 for t in tracks if t.is_confirmed()):
            survived_frames += 1
                
    return survived_frames, len(gt_trajectory)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    model = load_hardened_detector(cfg["paths"]["weights_out"])
    
    TEST_EPSILON = 0.1 
    print(f"[TEST] Evaluating with Epsilon: {TEST_EPSILON}")
    
    results = []
    for seq in cfg["data"]["eval_sequences"]:
        base_name = os.path.basename(seq)
        
        # Extract the base sequence name (e.g., MOT17-10) regardless of the suffix
        seq_key = base_name.replace("-FRCNN", "").replace("-Blackbox", "").replace("-Whitebox", "")
        
        target_id = TARGET_MAP.get(seq_key)
        
        # If the sequence isn't in our hardcoded map, auto-discover the longest trajectory
        if target_id is None:
            gt_file = os.path.join(seq, "gt", "gt.txt")
            if os.path.exists(gt_file):
                df = pd.read_csv(gt_file, header=None, names=["frame", "id", "x", "y", "w", "h", "active", "class", "vis"])
                pedestrians = df[df["class"] == 1]
                if not pedestrians.empty:
                    target_id = pedestrians['id'].value_counts().idxmax()
                    print(f"[*] Auto-selected Target ID {target_id} for {base_name} (Longest trajectory)")
                else:
                    print(f"[SKIP] No pedestrians found in {base_name}")
                    continue
            else:
                print(f"[SKIP] No gt.txt found for {base_name}")
                continue

        # Execute evaluation
        survived, total = evaluate_sequence(seq, model, target_id, epsilon=TEST_EPSILON)
        results.append(f"{base_name}: {survived}/{total} ({ (survived/total)*100:.1f}%)")
        print(results[-1])

    os.makedirs("outputs", exist_ok=True)
    log_path = f"outputs/robustness_eps_{str(TEST_EPSILON).replace('.', '_')}.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(results))
    print(f"[SUCCESS] Results saved to {log_path}")