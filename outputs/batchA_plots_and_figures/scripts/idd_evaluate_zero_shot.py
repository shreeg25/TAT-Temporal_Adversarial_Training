import os
import sys
import json
import xml.etree.ElementTree as ET
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Force path recognition
sys.path.insert(0, os.path.abspath("."))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESHOLD = 0.5
# IDD uses various classes. We need to specify what maps to your Class 1 (Pedestrian)
# Common IDD classes: 'pedestrian', 'person', 'rider'
TARGET_XML_CLASSES = ['pedestrian', 'person', 'rider'] 

def load_hardened_model(weight_path):
    print(f"[*] Loading ResNet-50 FPN on {DEVICE}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    print(f"[*] Loading hardened MNAT weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model

def parse_voc_xml(xml_path):
    """Parses a Pascal VOC XML file to extract target bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name in TARGET_XML_CLASSES:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
    return torch.tensor(bboxes, dtype=torch.float32).to(DEVICE)

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_dataset(img_dir, xml_dir, split_file, model):
    with open(split_file, 'r') as f:
        img_names = [line.strip() for line in f.readlines()]

    print(f"[*] Evaluating on {len(img_names)} frames from {split_file}...")
    
    total_true_positives = 0
    total_false_positives = 0
    total_ground_truths = 0

    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        xml_path = os.path.join(xml_dir, f"{img_name}.xml")
        
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
            
        gt_boxes = parse_voc_xml(xml_path)
        total_ground_truths += len(gt_boxes)

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            
        mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.5)
        pred_boxes = prediction['boxes'][mask]
        
        # Match predictions to GT using IoU
        matched_gt_indices = set()
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= IOU_THRESHOLD:
                total_true_positives += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                total_false_positives += 1 # The model hallucinated (e.g. detected a tree)
                
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{len(img_names)} images...")

    # Final Metrics
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-6)
    recall = total_true_positives / (total_ground_truths + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    print("\n" + "="*50)
    print("ZERO-SHOT IDD METRICS (Adversarial Smoothing Impact)")
    print("="*50)
    print(f"Total Ground Truth Pedestrians: {total_ground_truths}")
    print(f"True Positives (Correct Detections): {total_true_positives}")
    print(f"False Positives (Hallucinations/Trees): {total_false_positives}")
    print("-" * 50)
    print(f"Precision: {precision:.4f} (How many of its predictions were actually people?)")
    print(f"Recall:    {recall:.4f} (How many of the actual people did it find?)")
    print(f"F1:        {f1:.4f}")
    print("="*50)

    os.makedirs("outputs", exist_ok=True)
    metrics_out = {
        "total_ground_truths": total_ground_truths,
        "true_positives": total_true_positives,
        "false_positives": total_false_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": IOU_THRESHOLD,
    }
    with open("outputs/idd_zero_shot_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("[SAVED] outputs/idd_zero_shot_metrics.json")

if __name__ == "__main__":
    WEIGHT_PATH = "weights/tat_hardened_mnat_epoch_10.pth"
    BASE_DIR = "data/IDD_Detection"
    IMG_DIR = os.path.join(BASE_DIR, "JPEGImages")
    XML_DIR = os.path.join(BASE_DIR, "Annotations")
    SPLIT_FILE = os.path.join(BASE_DIR, "val.txt") # Using validation split for test
    
    model = load_hardened_model(WEIGHT_PATH)
    evaluate_dataset(IMG_DIR, XML_DIR, SPLIT_FILE, model)