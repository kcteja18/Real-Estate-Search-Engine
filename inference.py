"""
Floorplan Prediction Script

Loads a trained model and runs predictions on images from an Excel file.
This script is self-contained and does not require the training Config.
"""

import json
import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os

# ============================================================================
# 1. MODEL ARCHITECTURE (Required to re-create the model)
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
    Runs inference on a single floorplan image and returns a JSON string.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
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
        "bathrooms": predicted_counts.get('bathroom', 0)
    }
    
    return json.dumps(formatted_dict)


# ============================================================================
# 3. MAIN EXECUTION SCRIPT
# ============================================================================

if __name__ == '__main__':
    print("--- Floorplan Prediction Script (Lean, XLSX) ---")

    # --- SCRIPT CONFIGURATION ---
    TEST_IMAGE_DIR = r"D:\IITGN\Placement_prep\Real-Estate-Search-Engine\assets\images"
    MODEL_CHECKPOINT_PATH = r"D:\IITGN\Placement_prep\Real-Estate-Search-Engine\checkpoint_epoch_50.pth"
    INPUT_FILE = r"D:\IITGN\Placement_prep\Real-Estate-Search-Engine\assets\Property_list.xlsx"  # <-- Use the .xlsx file name
    OUTPUT_FILE = "Property_list_with_predictions.xlsx"
    
    # --- CRITICAL MODEL PARAMETERS (Hard-coded from training) ---
    MAX_ROOMS_PER_TYPE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if the test directory exists
    if not os.path.isdir(TEST_IMAGE_DIR):
        print(f"ERROR: The image directory '{TEST_IMAGE_DIR}' was not found.")
        sys.exit(1)

    # --- Load Model ---
    print(f"Loading model from {MODEL_CHECKPOINT_PATH}...")
    try:
        model = RoomDetectionCNN(max_rooms_per_type=MAX_ROOMS_PER_TYPE)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✓ Model loaded. Using device: {DEVICE.upper()}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # --- Load Input File ---
    print(f"Loading input file: {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE) 
        print(f"✓ Loaded {len(df)} properties.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)
    except ImportError:
        print("\nERROR: `openpyxl` library not found. It's needed to read .xlsx files.")
        print("Please install it: pip install openpyxl")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading input file: {e}")
        sys.exit(1)

    # --- Run Predictions ---
    print("Starting predictions...")
    prediction_json_list = []
    for index, row in df.iterrows():
        image_name = row.get('image_file')
        print(f"  Processing [{index+1}/{len(df)}]: {image_name}")

        if pd.isna(image_name) or image_name == "":
            print("    SKIPPED: No image file.")
            prediction_json_list.append(None)
            continue

        image_path = os.path.join(TEST_IMAGE_DIR, image_name)
        
        json_output = parse_floorplan(model, image_path, DEVICE)
        
        prediction_json_list.append(json_output)
        if json_output:
            print(f"    SUCCESS: {json_output}")

    print("✓ All predictions complete.")

    # --- Save Output File ---
    print("Adding JSON predictions to DataFrame...")
    df['floorplan_json'] = prediction_json_list
    
    try:
        print(f"Saving new file to: {OUTPUT_FILE}...")
        df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"\n✓ SUCCESS: New file created at {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nERROR saving output file: {e}")
        print("Please ensure 'openpyxl' is installed: pip install pandas openpyxl")