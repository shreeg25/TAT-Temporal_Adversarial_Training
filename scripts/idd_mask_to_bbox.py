import cv2
import numpy as np
import os
import json

def extract_bboxes_from_mask(mask_path, class_ids):
    """
    Reads an IDD segmentation mask and extracts bounding boxes for specific classes.
    """
    # Read the mask in grayscale mode. Do NOT scale or normalize.
    # We need the exact pixel values (0-30).
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Could not read mask: {mask_path}")
        return []

    bboxes = []
    
    # Iterate through the specific classes we care about (e.g., Pedestrian, Rider)
    for class_id in class_ids:
        # Create a binary mask where the pixels equal the target class ID
        binary_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)
        
        # Find the contours of the isolated class
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter out tiny noise artifacts (e.g., a 2x2 pixel patch)
            if cv2.contourArea(contour) < 50:
                continue
                
            # Extract the bounding box coordinates for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Format: [x_min, y_min, x_max, y_max, class_id]
            bboxes.append([x, y, x + w, y + h, class_id])
            
    return bboxes

def process_dataset(mask_dir, output_json_path):
    """
    Processes the entire mask archive and saves the bboxes to a JSON file.
    """
    # IDD Segmentation Mapping (Level 3 Hierarchy)
    # 11: Person (Pedestrian)
    # 12: Rider
    TARGET_CLASSES = [11, 12] 
    
    dataset_annotations = {}
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
    print(f"[*] Processing {len(mask_files)} masks from {mask_dir}...")
    
    for mask_name in mask_files:
        mask_path = os.path.join(mask_dir, mask_name)
        
        # The image name usually matches the mask name, replacing '_label.png' with '.jpg'
        # Adjust this split logic depending on your exact IDD file naming convention
        base_name = mask_name.split('_label')[0] if '_label' in mask_name else mask_name.split('.')[0]
        
        bboxes = extract_bboxes_from_mask(mask_path, TARGET_CLASSES)
        
        if bboxes:
            dataset_annotations[base_name] = bboxes

    # Save to a clean JSON file
    with open(output_json_path, 'w') as f:
        json.dump(dataset_annotations, f, indent=4)
        
    print(f"[SUCCESS] Extracted bounding boxes saved to {output_json_path}")
    print(f"[*] Total frames with pedestrians/riders: {len(dataset_annotations)}")

if __name__ == "__main__":
    # Ensure these paths match your local directory structure
    MASK_ARCHIVE_DIR = "data/IDD_RESIZED/mask_archive" 
    OUTPUT_ANNOTATIONS = "data/IDD_RESIZED/idd_bbox_annotations.json"
    
    os.makedirs(os.path.dirname(OUTPUT_ANNOTATIONS), exist_ok=True)
    process_dataset(MASK_ARCHIVE_DIR, OUTPUT_ANNOTATIONS)