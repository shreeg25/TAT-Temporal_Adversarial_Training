import os
import sys
import torch
import random
import yaml

# ==============================================================================
# ENVIRONMENT PATH INJECTION
# ==============================================================================
# Forces Python to recognize the root directory so it can find the 'src' module
sys.path.insert(0, os.path.abspath("."))

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset import MOT17Dataset
from src.device import DEVICE

# ==============================================================================
# MULTI-NORM ADVERSARIAL GENERATORS
# ==============================================================================

def pgd_attack_linf(images, targets, model, eps, alpha=2/255, iters=4):
    """L_inf PGD: Perturbs all pixels by a tiny maximum threshold."""
    adv_images = [img.clone().detach().to(DEVICE) for img in images]
    for img in adv_images:
        img.requires_grad = True

    model.train()
    for _ in range(iters):
        loss_dict = model(adv_images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.zero_grad()
        losses.backward()

        with torch.no_grad():
            for i in range(len(adv_images)):
                grad_sign = adv_images[i].grad.sign()
                adv_images[i] = adv_images[i] + alpha * grad_sign
                # Project back to L_inf epsilon ball
                eta = torch.clamp(adv_images[i] - images[i], min=-eps, max=eps)
                adv_images[i] = torch.clamp(images[i] + eta, min=0, max=1)
                adv_images[i].requires_grad = True
                
    return [img.detach() for img in adv_images]

def pgd_attack_l2(images, targets, model, eps, alpha=0.5, iters=4):
    """L_2 PGD: Smooth, Euclidean energy-bounded perturbations."""
    adv_images = [img.clone().detach().to(DEVICE) for img in images]
    for img in adv_images:
        img.requires_grad = True

    model.train()
    for _ in range(iters):
        loss_dict = model(adv_images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.zero_grad()
        losses.backward()

        with torch.no_grad():
            for i in range(len(adv_images)):
                grad = adv_images[i].grad
                # L2 Normalization of the gradient
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), p=2, dim=1).reshape(-1, 1, 1)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                normalized_grad = grad / grad_norm
                
                adv_images[i] = adv_images[i] + alpha * normalized_grad
                
                # Project back to L_2 epsilon ball
                eta = adv_images[i] - images[i]
                eta_norm = torch.norm(eta.reshape(eta.shape[0], -1), p=2, dim=1).reshape                                                                (-1, 1, 1)
                factor = torch.min(torch.tensor(1.0).to(DEVICE), eps / (eta_norm + 1e-8))
                
                adv_images[i] = torch.clamp(images[i] + eta * factor, min=0, max=1)
                adv_images[i].requires_grad = True
                
    return [img.detach() for img in adv_images]

def sparse_l1_attack(images, targets, model, eps, sparsity_ratio=0.05, iters=4):
    """L_1 Approximation (Top-K Sparse): Simulates a high-contrast physical sticker."""
    adv_images = [img.clone().detach().to(DEVICE) for img in images]
    for img in adv_images:
        img.requires_grad = True

    model.train()
    for _ in range(iters):
        loss_dict = model(adv_images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.zero_grad()
        losses.backward()

        with torch.no_grad():
            for i in range(len(adv_images)):
                grad = adv_images[i].grad
                grad_abs = torch.abs(grad)
                
                # Calculate the Top-K threshold for the sparsity mask
                k = int(sparsity_ratio * grad.numel())
                threshold = torch.kthvalue(grad_abs.reshape(-1), grad.numel() - k)[0]
                
                # Mask out 95% of the gradients, only update the most critical pixels
                sparse_mask = (grad_abs >= threshold).float()
                adv_images[i] = adv_images[i] + (eps * grad.sign() * sparse_mask)
                
                # Clip strictly to image bounds (no epsilon ball projection needed for L0/L1 mask approach)
                adv_images[i] = torch.clamp(adv_images[i], min=0, max=1)
                adv_images[i].requires_grad = True
                
    return [img.detach() for img in adv_images]

def generate_multinorm_patch(images, targets, model, epsilon_dict):
    """Stochastic Alternating Norm router."""
    attack_type = random.choice(['L_inf', 'L_2', 'L_1'])
    
    if attack_type == 'L_inf':
        return pgd_attack_linf(images, targets, model, eps=epsilon_dict['L_inf'])
    elif attack_type == 'L_2':
        return pgd_attack_l2(images, targets, model, eps=epsilon_dict['L_2'])
    elif attack_type == 'L_1':
        return sparse_l1_attack(images, targets, model, eps=epsilon_dict['L_1'])

# ==============================================================================
# ARCHITECTURE & TRAINING LOOP
# ==============================================================================

def unfreeze_backbone(model):
    """Forcefully unfreezes the ResNet backbone. Mandatory for Adversarial Defense."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("[WARN] ResNet50 Backbone completely unfrozen for Adversarial Regularization.")

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    
    # 1. Initialize Datasets
    print("[INIT] Loading MOT17 Datasets...")
    train_dataset = MOT17Dataset(cfg['paths']['mot17_train'])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # 2. Load Pre-trained Baseline
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2) # Pedestrian + Background
    
    # CRITICAL: We must unfreeze the backbone or the defense fails mathematically.
    unfreeze_backbone(model)
    model.to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 3. Calibrated Epsilon Bounds for Normalized Tensors [0, 1]
    epsilons = {
        'L_inf': 8 / 255.0, # Invisible distributed noise
        'L_2': 1.5,         # Smooth Euclidean boundary
        'L_1': 0.1          # 10% magnitude shift on 5% of pixels (Sticker effect)
    }
    
    print("[TRAIN] Initiating Multi-Norm Adversarial Training (MNAT) Crucible...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            # Format inputs
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Layer 1 Defense: Generate alternating adversarial patches on the fly
            adv_images = generate_multinorm_patch(images, targets, model, epsilons)
            
            # Clear optimizer
            optimizer.zero_grad()
            
            # Forward pass on the multi-norm poisoned data
            loss_dict = model(adv_images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass & Optimize
            losses.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {losses.item():.4f}")
                
        # Save hardened weights per epoch
        torch.save(model.state_dict(), f"weights/tat_hardened_mnat_epoch_{epoch+1}.pth")

    print("[SUCCESS] Multi-Norm training complete. Network is hardened against L_inf, L_2, and L_1 manifolds.")

if __name__ == "__main__":
    main()