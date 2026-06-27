import sys
import os
import yaml
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.insert(0, os.path.abspath("."))
from src.device import DEVICE

def load_hardened_detector(weight_path):
    print(f"[EXPORT] Loading TAT weights from {weight_path}...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    detector = load_hardened_detector(cfg["paths"]["weights_out"])
    
    # 1. Define target directory exactly where TrackEval expects it
    # We use 'TAT_MNAT_Architecture' to match the folder you renamed
    tracker_name = "TAT_MNAT_Architecture"
    eval_dir = os.path.join("TrackEval", "data", "trackers", "mot_challenge", "MOT17-train", tracker_name, "data")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 2. Dynamically fetch sequences from config
    sequences_to_export = cfg["data"]["eval_sequences"]
    
    for seq_path in sequences_to_export:
        if not os.path.exists(seq_path):
            print(f"[SKIP] Path not found: {seq_path}")
            continue
            
        # 3. Determine output filename based on folder name
        base_name = os.path.basename(seq_path)
        if "Whitebox" in base_name or "Blackbox" in base_name:
            out_name = f"{base_name}.txt"
        else:
            out_name = f"{base_name}-Clean.txt"
            
        output_txt_path = os.path.join(eval_dir, out_name)
        print(f"[EXPORT] Processing {base_name} -> {output_txt_path}")
        
        # 4. Tracking Inference
        tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=(DEVICE.type=='cuda'))
        img_dir = os.path.join(seq_path, "img1")
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        with open(output_txt_path, "w") as out_file:
            for frame_name in frames:
                frame_idx = int(frame_name.split('.')[0])
                img_path = os.path.join(img_dir, frame_name)
                bgr = cv2.imread(img_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(DEVICE)
                with torch.no_grad():
                    preds = detector([tensor])[0]
                    
                labels, scores, boxes = preds["labels"].cpu().numpy(), preds["scores"].cpu().numpy(), preds["boxes"].cpu().numpy()
                keep = (labels == 1) & (scores > 0.4)
                boxes, scores = boxes[keep], scores[keep]
                
                xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for (x1, y1, x2, y2) in boxes]
                raw_dets = [[d, float(s), "1"] for d, s in zip(xywh, scores)]
                
                tracks = tracker.update_tracks(raw_dets, frame=rgb)
                
                for t in tracks:
                    if not t.is_confirmed(): continue
                    ltrb = t.to_ltrb()
                    x, y, w, h = ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
                    out_file.write(f"{frame_idx},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                    
    print(f"[SUCCESS] Metrics exported to {eval_dir}")