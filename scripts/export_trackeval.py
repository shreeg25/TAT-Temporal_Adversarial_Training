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
    
    # We will export the Clean and Whitebox datasets for MOT17-04
    # TrackEval requires the output filename to EXACTLY match the sequence name in the GT folder.
    sequences_to_export = [
        ("data/MOT17/train/MOT17-04-FRCNN", "MOT17-04-FRCNN-Clean"),
        ("data/MOT17/train/MOT17-04-FRCNN-Whitebox", "MOT17-04-FRCNN-Whitebox")
    ]
    
    # The painful, strict directory structure TrackEval requires
    tracker_name = "TAT_Architecture"
    eval_dir = f"TrackEval/data/trackers/mot_challenge/MOT17-train/{tracker_name}/data"
    os.makedirs(eval_dir, exist_ok=True)
    
    for seq_path, output_name in sequences_to_export:
        if not os.path.exists(seq_path):
            continue
            
        tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=(DEVICE.type=='cuda'))
        img_dir = os.path.join(seq_path, "img1")
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        output_txt_path = os.path.join(eval_dir, f"{output_name}.txt")
        print(f"[EXPORT] Processing {output_name} -> {output_txt_path}")
        
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
                    # Convert to MOT17 required bounding box format (x_top_left, y_top_left, width, height)
                    x, y, w, h = ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
                    
                    # Formatting: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    out_file.write(f"{frame_idx},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                    
    print("[SUCCESS] All tracking data formatted and injected into the TrackEval directory.")