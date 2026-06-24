# scripts/evaluate_tat.py
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

# THE BRIDGE
sys.path.insert(0, os.path.abspath("."))
from src.device import DEVICE

# The specific vulnerable targets you are tracking
TARGET_MAP = {
    "MOT17-02": 19,
    "MOT17-04": 1,
    "MOT17-09": 12
}

def load_hardened_detector(weight_path):
    print(f"[EVAL] Loading TAT-Hardened Feature Pyramid from {weight_path}...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def evaluate_sequence(seq_path, model, target_id):
    img_dir = os.path.join(seq_path, "img1")
    if not os.path.exists(img_dir):
        return 0, 0

    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=(DEVICE.type=='cuda'))
    
    total_frames = len(frames)
    survived_frames = 0
    
    for frame_name in frames:
        img_path = os.path.join(img_dir, frame_name)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Detector Inference
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(DEVICE)
        preds = model([tensor])[0]
        
        labels, scores, boxes = preds["labels"].cpu().numpy(), preds["scores"].cpu().numpy(), preds["boxes"].cpu().numpy()
        keep = (labels == 1) & (scores > 0.4)
        boxes, scores = boxes[keep], scores[keep]
        
        # Format for DeepSORT: [x, y, w, h]
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for (x1, y1, x2, y2) in boxes]
        raw_dets = [[d, float(s), "1"] for d, s in zip(xywh, scores)]
        
        tracks = tracker.update_tracks(raw_dets, frame=rgb)
        
        # Check if our vulnerable target survived this frame
        for t in tracks:
            if t.is_confirmed() and t.track_id == str(target_id):
                survived_frames += 1
                break
                
    return survived_frames, total_frames

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    eval_seqs = cfg["data"]["eval_sequences"]
    
    detector = load_hardened_detector(cfg["paths"]["weights_out"])
    
    print("\n" + "="*85)
    print(f"{'TAT ARCHITECTURE: TARGET SURVIVAL METRICS':^85}")
    print("="*85)
    print(f"{'Sequence':<30} | {'Target ID':<10} | {'Survived / Total':<20} | {'Survival Rate':<15}")
    print("-" * 85)
    
    for seq in eval_seqs:
        if not os.path.exists(seq): continue
        
        # Extract base sequence name to find the correct target ID
        base_name = os.path.basename(seq)
        seq_key = base_name.split("-FRCNN")[0] 
        target_id = TARGET_MAP.get(seq_key, None)
        
        if target_id is None: continue
        
        survived, total = evaluate_sequence(seq, detector, target_id)
        rate = (survived / total) * 100 if total > 0 else 0
        
        print(f"{base_name:<30} | {target_id:<10} | {f'{survived}/{total}':<20} | {rate:>5.1f}%")
        
    print("="*85)