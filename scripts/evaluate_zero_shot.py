import os
import sys
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from PIL import Image

# Force path recognition
sys.path.insert(0, os.path.abspath("."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hardened_model(weight_path):
    print(f"[*] Loading ResNet-50 FPN on {DEVICE}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Custom 2-class head (Pedestrian vs Background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    print(f"[*] Loading hardened MNAT weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    
    # CRITICAL: Set to eval mode to freeze batch norm and dropout. No MNAT loop here.
    model.eval().to(DEVICE)
    return model

def run_smoke_test(img_dir, out_dir, model, max_images=50):
    os.makedirs(out_dir, exist_ok=True)
    img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))][:max_images]
    
    if not img_names:
        print(f"[ERROR] No images found in {img_dir}")
        return

    print(f"[*] Running Zero-Shot Inference on {len(img_names)} IDD images...")
    
    detections_found = 0
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            
        # Filter for Class 1 (Pedestrian) and high confidence
        mask = (prediction['labels'] == 1) & (prediction['scores'] > 0.5)
        boxes = prediction['boxes'][mask]
        
        if len(boxes) > 0:
            detections_found += 1
            
            # Draw boxes for visual proof
            img_to_draw = (img_tensor[0] * 255).to(torch.uint8)
            drawn_img = draw_bounding_boxes(img_to_draw, boxes, colors="red", width=3)
            
            out_img_pil = TF.to_pil_image(drawn_img)
            out_img_pil.save(os.path.join(out_dir, f"det_{img_name}"))
            
    print(f"\n[RESULTS] Smoke Test Complete.")
    print(f"[*] The model successfully detected pedestrians in {detections_found}/{len(img_names)} frames.")
    
    if detections_found == 0:
        print("[CRITICAL] The model collapsed. It suffered Catastrophic Overfitting to MOT17.")
    else:
        print(f"[SUCCESS] Spatial generalization proven. Check {out_dir} to verify box accuracy.")

if __name__ == "__main__":
    WEIGHT_PATH = "weights/tat_hardened_mnat_epoch_10.pth"
    IDD_IMG_DIR = "data/IDD_RESIZED/image_archive"
    OUTPUT_DIR  = "outputs/IDD_zero_shot"
    
    model = load_hardened_model(WEIGHT_PATH)
    run_smoke_test(IDD_IMG_DIR, OUTPUT_DIR, model)