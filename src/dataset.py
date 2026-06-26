import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class MOT17Dataset(Dataset):
    def __init__(self, root_dir):
        """
        Parses the MOTChallenge directory tree and strictly formats 
        ground truth bounding boxes for Faster R-CNN ingestion.
        """
        self.root_dir = root_dir
        self.frames = []
        
        print(f"[DATA] Crawling MOT17 directory: {self.root_dir}...")
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"CRITICAL: Dataset path {root_dir} does not exist.")

        # Iterate through sequence folders
        for seq in os.listdir(root_dir):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            # =================================================================
            # STRICT FIREWALL: Ignore pre-generated evaluation datasets
            # =================================================================
            if "Whitebox" in seq or "Blackbox" in seq or "Clean" in seq:
                print(f"  -> [FILTER] Ignoring pre-generated adversarial sequence: {seq}")
                continue

            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            img_dir = os.path.join(seq_path, 'img1')
            
            if not os.path.exists(gt_path) or not os.path.exists(img_dir):
                print(f"  -> [WARN] Skipping {seq} (Missing gt.txt or img1 directory)")
                continue

            # Parse the ground truth matrix
            seq_gt = {}
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])
                    class_id = int(parts[7])
                    visibility = float(parts[8])

                    # FILTERING: Pedestrians (Class 1) with >= 20% visibility
                    if class_id != 1 or visibility < 0.2:
                        continue

                    # MOT format (x, y, w, h) -> PyTorch format (x1, y1, x2, y2)
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    x2 = x1 + float(parts[4])
                    y2 = y1 + float(parts[5])
                    
                    # Bounds checking to prevent C++ backend crashes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    if frame_id not in seq_gt:
                        seq_gt[frame_id] = []
                    seq_gt[frame_id].append([x1, y1, x2, y2])

            valid_frames = 0
            for frame_name in sorted(os.listdir(img_dir)):
                if not frame_name.endswith('.jpg'): continue
                
                frame_id = int(frame_name.split('.')[0])
                img_path = os.path.join(img_dir, frame_name)
                boxes = seq_gt.get(frame_id, [])
                
                # Drop empty frames to prevent NaN loss
                if len(boxes) > 0:
                    self.frames.append((img_path, boxes))
                    valid_frames += 1
            
            print(f"  -> Mapped {valid_frames} clean, valid frames from {seq}")
            
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path, boxes = self.frames[idx]

        # 1. Load and format the image tensor
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Corrupt image file: {img_path}")
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and permute to [C, H, W] for PyTorch
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        # 2. Format the target dictionary
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        # All valid boxes are pedestrians, so class label is always 1
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target