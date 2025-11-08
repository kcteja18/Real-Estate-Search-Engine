"""
Floorplan Model Definition and Inference Function

Contains the RoomDetectionCNN model architecture and the
parse_floorplan function for running inference on a single image.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

# ============================================================================
# 1. MODEL ARCHITECTURE
# ============================================================================

class RoomDetectionCNN(nn.Module):
    """CNN for room detection in floorplans"""

    def __init__(self, max_rooms_per_type=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False) 
        self.backbone.fc = nn.Identity()
        self.fc_shared = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3)
        )
        self.room_heads = nn.ModuleDict({
            'bedroom': nn.Linear(128, max_rooms_per_type),
            'kitchen': nn.Linear(128, max_rooms_per_type),
            'bathroom': nn.Linear(128, max_rooms_per_type),
            'garage': nn.Linear(128, max_rooms_per_type),
            'hall': nn.Linear(128, max_rooms_per_type)
        })
        self.total_rooms_head = nn.Linear(128, max_rooms_per_type * 2)

    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.fc_shared(features)
        predictions = {}
        for room_type in ['bedroom', 'kitchen', 'bathroom', 'garage', 'hall']:
            predictions[room_type] = self.room_heads[room_type](shared_features)
        predictions['total_rooms'] = self.total_rooms_head(shared_features)
        return predictions


# ============================================================================
# 2. PREDICTION FUNCTION
# ============================================================================

def parse_floorplan(model, image_path, device):
    """
    Runs inference on a single floorplan image and returns a dict.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        # Use standard ImageNet stats as the model backbone is ResNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
    except FileNotFoundError:
        print(f"    WARNING: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"    ERROR: Could not process image {image_path}: {e}")
        return None

    with torch.no_grad():
        predictions_raw = model(input_tensor)

    predicted_counts = {}
    for room_type in ['bedroom', 'kitchen', 'bathroom', 'garage', 'hall']:
        logits = predictions_raw[room_type][0]
        pred_count = logits.argmax().item()
        predicted_counts[room_type] = pred_count
        
    formatted_dict = {
        "rooms": predicted_counts.get('bedroom', 0),
        "halls": predicted_counts.get('hall', 0),
        "kitchens": predicted_counts.get('kitchen', 0),
        "bathrooms": predicted_counts.get('bathroom', 0),
        "garages": predicted_counts.get('garage', 0) 
    }
    
    return formatted_dict