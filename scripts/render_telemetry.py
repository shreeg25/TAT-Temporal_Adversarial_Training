import sys
import os
import yaml
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, os.path.abspath("."))
from src.device import DEVICE

def load_hardened_detector(weight_path):
    print(f"[RENDER] Loading TAT weights from {weight_path}...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    
    # We explicitly render the Whitebox attack to prove the defense visually
    seq_path = "data/MOT17/train/MOT17-04-FRCNN-Whitebox"
    out_dir = "outputs/render_frames"
    os.makedirs(out_dir, exist_ok=True)
    
    detector = load_hardened_detector(cfg["paths"]["weights_out"])
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=(DEVICE.type=='cuda'))
    
    img_dir = os.path.join(seq_path, "img1")
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    print(f"[RENDER] Processing frames from {seq_path}...")
    
    # Render just the first 50 frames to save your GPU time and get the screenshots you need
    for frame_name in frames[:50]:
        img_path = os.path.join(img_dir, frame_name)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Inference
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(DEVICE)
        with torch.no_grad():
            preds = detector([tensor])[0]
            
        labels, scores, boxes = preds["labels"].cpu().numpy(), preds["scores"].cpu().numpy(), preds["boxes"].cpu().numpy()
        keep = (labels == 1) & (scores > 0.4)
        boxes, scores = boxes[keep], scores[keep]
        
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for (x1, y1, x2, y2) in boxes]
        raw_dets = [[d, float(s), "1"] for d, s in zip(xywh, scores)]
        
        tracks = tracker.update_tracks(raw_dets, frame=rgb)
        
        # Draw colorful bounding boxes
        for t in tracks:
            if not t.is_confirmed(): continue
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Bright Neon Green for TAT Tracking
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Magenta ID tag
            cv2.putText(bgr, f"ID: {t.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        out_path = os.path.join(out_dir, frame_name)
        cv2.imwrite(out_path, bgr)
        
    print(f"[SUCCESS] Visual telemetry rendered with colorful bounding boxes to {out_dir}")