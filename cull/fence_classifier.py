"""
fence_classifier.py — Wire fence detection inference.

Loads pretrained fence binary classifier and provides inference functions.
Models: ResNet18 (fastest), MobileNetV2 (best F1), ResNet50 (balanced)

Usage:
    classifier = FenceClassifier(arch='mobilenetv2')
    pred_binary, confidence = classifier.predict_image(img_path)  # 0 or 1, [0, 1]
    batch_preds = classifier.predict_batch(img_paths)              # array of 0/1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FenceClassifier:
    """Wire fence binary classifier."""
    
    def __init__(self, arch: str = "mobilenetv2", checkpoint_dir: str = "fence_classifier_checkpoints"):
        """
        Initialize fence classifier.
        
        Args:
            arch: Model architecture ('resnet18', 'resnet50', 'mobilenetv2')
            checkpoint_dir: Directory containing model checkpoints
        """
        self.arch = arch
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = self._build_and_load_model()
        self.model.eval()
        
        # Preprocessing transform (must match training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        log.info(f"FenceClassifier initialized: {arch}, device={device}")
    
    def _build_and_load_model(self) -> nn.Module:
        """Build model architecture and load weights."""
        # Build model
        if self.arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)
        elif self.arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)
        elif self.arch == "mobilenetv2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unknown arch: {self.arch}")
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / self.arch / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model
    
    def predict_image(self, img_path: Union[str, Path]) -> tuple[int, float]:
        """
        Predict fence class for a single image.
        
        Args:
            img_path: Path to image file
        
        Returns:
            (pred_class, confidence)
            - pred_class: 0 (no fence) or 1 (fence)
            - confidence: probability of predicted class [0, 1]
        """
        img_path = Path(img_path)
        
        # Load image
        try:
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning(f"Failed to load {img_path}, returning pred=0")
                return 0, 0.0
        except Exception as e:
            log.warning(f"Error loading {img_path}: {e}, returning pred=0")
            return 0, 0.0
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = self.model(img_tensor).squeeze()
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob > 0.5 else 0
        
        return pred, prob
    
    def predict_batch(self, img_paths: list[Union[str, Path]], batch_size: int = 32) -> np.ndarray:
        """
        Predict fence class for a batch of images.
        
        Args:
            img_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            Array of predictions (0 or 1) with same length as img_paths
        """
        preds = []
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            batch_imgs = []
            
            for img_path in batch_paths:
                try:
                    img = cv2.imdecode(
                        np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is None:
                        batch_imgs.append(np.zeros((3, 224, 224), dtype=np.float32))
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img)
                    batch_imgs.append(img_tensor)
                except Exception as e:
                    log.warning(f"Error loading {img_path}: {e}")
                    batch_imgs.append(torch.zeros(3, 224, 224))
            
            if batch_imgs:
                batch_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    logits = self.model(batch_tensor).squeeze()
                    probs = torch.sigmoid(logits)
                    batch_preds = (probs > 0.5).float().cpu().numpy()
                preds.extend(batch_preds)
        
        return np.array(preds, dtype=np.int32)
    
    def predict_with_confidence(
        self, img_paths: list[Union[str, Path]], batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict fence class and confidence for a batch of images.
        
        Args:
            img_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            (preds, confidences)
            - preds: Array of predictions (0 or 1)
            - confidences: Array of confidence scores [0, 1]
        """
        preds = []
        confidences = []
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            batch_imgs = []
            
            for img_path in batch_paths:
                try:
                    img = cv2.imdecode(
                        np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is None:
                        batch_imgs.append(torch.zeros(3, 224, 224))
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img)
                    batch_imgs.append(img_tensor)
                except Exception as e:
                    log.warning(f"Error loading {img_path}: {e}")
                    batch_imgs.append(torch.zeros(3, 224, 224))
            
            if batch_imgs:
                batch_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    logits = self.model(batch_tensor).squeeze()
                    probs = torch.sigmoid(logits)
                    batch_preds = (probs > 0.5).float().cpu().numpy()
                    batch_confs = probs.cpu().numpy()
                preds.extend(batch_preds)
                confidences.extend(batch_confs)
        
        return np.array(preds, dtype=np.int32), np.array(confidences, dtype=np.float32)
