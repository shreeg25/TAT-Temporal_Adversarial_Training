# scripts/train_baseline.py
import sys
import os
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

class CleanMOTDataset(Dataset):
    def __init__(self, clean_seqs):
        self.samples = []
        for clean_seq in clean_seqs:
            img_dir = os.path.join(clean_seq, "img1")
            gt_file = os.path.join(clean_seq, "gt", "gt.txt")
            if not os.path.exists(img_dir) or not os.path.exists(gt_file): continue
            
            df = pd.read_csv(gt_file, header=None, names=["frame", "id", "x", "y", "w", "h", "active", "class", "vis"])
            df = df[(df["active"] == 1) & (df["class"] == 1) & (df["vis"] >= 0.25)]
            
            for frame_no, grp in df.groupby("frame"):
                img_path = os.path.join(img_dir, f"{int(frame_no):06d}.jpg")
                boxes = grp[["x", "y", "w", "h"]].values
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                self.samples.append((img_path, boxes.astype(np.float32)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }
        return tensor_img, target

def collate_fn(batch): return tuple(zip(*batch))

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    clean_seqs = cfg["data"].get("train_sequences", [])
    
    dataset = CleanMOTDataset(clean_seqs)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(DEVICE)
    
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    model.train()
    for epoch in range(2): # 2 epochs is enough for standard domain adaptation
        for i, (images, targets) in enumerate(loader):
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if i % 50 == 0: print(f"Baseline Forge | Epoch {epoch+1}/2 | Batch {i} | Loss: {losses.item():.4f}")
            
    os.makedirs("weights", exist_ok=True)
    # Save to a DIFFERENT file so we don't destroy your TAT weights
    save_path = "weights/faster_rcnn_mot17_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Naive control group saved to {save_path}")