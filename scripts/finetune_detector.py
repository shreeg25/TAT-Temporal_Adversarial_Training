# scripts/finetune_detector.py
import sys
import os
import random

# THE BRIDGE: Force Python to recognize the root directory
sys.path.insert(0, os.path.abspath("."))

import torch
import pandas as pd
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.device import DEVICE

class DualStreamMOTDataset(Dataset):
    def __init__(self, clean_seqs, poison_prob=0.5):
        """
        TAT Dual-Stream Loader
        Randomly samples from clean or poisoned streams, but ALWAYS anchors
        the loss calculation to the clean ground truth trajectories.
        """
        self.samples = []
        self.poison_prob = poison_prob
        
        for clean_seq in clean_seqs:
            img_dir_clean = os.path.join(clean_seq, "img1")
            if not os.path.exists(img_dir_clean): continue
            
            # Identify the corresponding poisoned sequence
            poison_seq = f"{clean_seq}-Whitebox"
            img_dir_poison = os.path.join(poison_seq, "img1")
            has_poison = os.path.exists(img_dir_poison)
            
            # Load Clean Ground Truth ALWAYS
            gt_file = os.path.join(clean_seq, "gt", "gt.txt")
            if not os.path.exists(gt_file): continue
            
            df = pd.read_csv(gt_file, header=None, names=["frame", "id", "x", "y", "w", "h", "active", "class", "vis"])
            # Filter for active, visible pedestrians
            df = df[(df["active"] == 1) & (df["class"] == 1) & (df["vis"] >= 0.25)]
            
            for frame_no, grp in df.groupby("frame"):
                filename = f"{int(frame_no):06d}.jpg"
                clean_path = os.path.join(img_dir_clean, filename)
                poison_path = os.path.join(img_dir_poison, filename) if has_poison else clean_path
                
                boxes = grp[["x", "y", "w", "h"]].values
                # Convert xywh to xyxy for torchvision
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                
                self.samples.append((clean_path, poison_path, boxes.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_path, poison_path, boxes = self.samples[idx]
        
        # ── THE DUAL STREAM (Decoupled Input) ──
        if random.random() < self.poison_prob and os.path.exists(poison_path):
            img_path = poison_path
        else:
            img_path = clean_path
            
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch format [C, H, W] and normalize to [0, 1]
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # ── THE GROUND TRUTH ANCHOR (Decoupled Output) ──
        target = {}
        target["boxes"] = torch.from_numpy(boxes)
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64) # Label 1 is Pedestrian
        
        return tensor_img, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    
    # Fetch sequences directly from the updated config's phase 1 block
    clean_seqs = cfg["data"].get("train_sequences", [])
    
    print(f"[FINETUNE] Assembling TAT Dual-Stream dataset from {len(clean_seqs)} sequences...")
    dataset = DualStreamMOTDataset(clean_seqs, poison_prob=0.5)
    
    # Use batch size 2 and num_workers 0 strictly to protect your 8GB VRAM
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    print("[FINETUNE] Downloading base COCO weights and injecting 2-Class Head...")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    EPOCHS = 3  # Increased slightly so the network learns to ignore the patch
    
    print("[FINETUNE] Initiating Temporal Adversarial Training (TAT)...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (images, targets) in enumerate(loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(loader)} | Loss: {losses.item():.4f}")
                
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(loader):.4f} ===")
        
    os.makedirs("weights", exist_ok=True)
    # Overwrite the original weights so the rest of the pipeline automatically uses the hardened brain
    save_path = "weights/faster_rcnn_mot17.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[FINETUNE] Success. TAT hardened domain weights locked and saved to {save_path}")