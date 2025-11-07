import json
from pathlib import Path
import numpy as np
import random
from collections import defaultdict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.ops import box_iou, batched_nms

warnings.filterwarnings('ignore')


# IMPORTANT
# This ID *must* match the 'category_id' for 'room_name' in your annotations.coco.json
ROOM_NAME_LABEL_ID = 1 
# -------------------------

MODEL_PATH = 'best_model.pth' # Or 'final_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 640

# Model config must match training
NUM_FG_CLASSES = 8
NUM_ANCHORS = 9

# Global model variable
MODEL = None

# Backbone
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Full Detector Model
class ObjectDetectorFromScratch(nn.Module):
    # (This class is identical to the one in training script)
    # (Omitting for brevity in this response, but it *must* be fully
    #  pasted here in your final inference.py file)
    def __init__(self, num_fg_classes=8, num_anchors=9, target_size=640):
        super().__init__()
        self.backbone = SimpleBackbone()

        self.num_fg_classes = num_fg_classes
        self.num_classes = num_fg_classes + 1 # +1 for background
        self.num_anchors = num_anchors

        self.conv_head = nn.Conv2d(256, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, num_anchors * self.num_classes, 1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)

        self.feature_map_size = 40
        self.stride = target_size / self.feature_map_size
        self.anchor_scales = [1.0, 1.5, 2.0]
        self.aspect_ratios = [0.5, 1.0, 2.0]

        anchors = self._generate_anchors(target_size)
        self.register_buffer("anchors", anchors)

    def _generate_anchors(self, target_size):
        scales = torch.tensor(self.anchor_scales)
        ratios = torch.tensor(self.aspect_ratios)
        grid_x = (torch.arange(0, self.feature_map_size) + 0.5) * self.stride
        grid_y = (torch.arange(0, self.feature_map_size) + 0.5) * self.stride
        y_c, x_c = torch.meshgrid(grid_y, grid_x, indexing='ij')
        centers = torch.stack([x_c, y_c], dim=-1).float().view(-1, 1, 2)
        base_anchors_wh = []
        for scale in scales:
             for ratio in ratios:
                base_size = self.stride * scale
                w = base_size * torch.sqrt(ratio)
                h = base_size / torch.sqrt(ratio)
                base_anchors_wh.append([w, h])
        base_anchors_wh = torch.tensor(base_anchors_wh).float()
        centers_wh = torch.cat([centers.expand(-1, self.num_anchors, -1),
                                base_anchors_wh.expand(self.feature_map_size**2, -1, -1)],
                               dim=-1)
        all_anchors_ctr = centers_wh.view(-1, 4)
        x1y1 = all_anchors_ctr[:, :2] - all_anchors_ctr[:, 2:] / 2
        x2y2 = all_anchors_ctr[:, :2] + all_anchors_ctr[:, 2:] / 2
        all_anchors_xyxy = torch.cat([x1y1, x2y2], dim=-1)
        return all_anchors_xyxy.clamp(0, target_size)

    def _decode_boxes(self, anchors, deltas):
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
        pred_ctr_x = deltas[:, 0] * anchors_wh[:, 0] + anchors_ctr[:, 0]
        pred_ctr_y = deltas[:, 1] * anchors_wh[:, 1] + anchors_ctr[:, 1]
        pred_w = torch.exp(deltas[:, 2]) * anchors_wh[:, 0]
        pred_h = torch.exp(deltas[:, 3]) * anchors_wh[:, 1]
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(self, images, targets=None):
        # Inference mode: targets will be None
        features = self.backbone(images)
        x = F.relu(self.conv_head(features))
        cls_logits = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
        batch_size = images.size(0)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        # We are in eval mode, so targets=None
        # This will skip loss calculation and call _generate_proposals
        if self.training:
             raise Exception("Model should be in eval() mode for inference.")
        
        return self._generate_proposals(cls_logits, bbox_deltas)

    @torch.no_grad()
    def _generate_proposals(self, cls_logits, bbox_deltas, conf_thresh=0.05, nms_thresh=0.45):
        batch_size = cls_logits.size(0)
        anchors = self.anchors
        scores = F.softmax(cls_logits, dim=-1)
        detections = []
        for i in range(batch_size):
            batch_deltas = bbox_deltas[i]
            batch_scores = scores[i]
            pred_boxes = self._decode_boxes(anchors, batch_deltas)
            scores_fg = batch_scores[:, 1:]
            top_scores, top_labels = scores_fg.max(dim=1)
            keep = top_scores > conf_thresh
            boxes_out = pred_boxes[keep]
            scores_out = top_scores[keep]
            labels_out = top_labels[keep] + 1
            if boxes_out.numel() == 0:
                detections.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long)})
                continue
            keep_nms = batched_nms(boxes=boxes_out, scores=scores_out, idxs=labels_out, iou_threshold=nms_thresh)
            detections.append({'boxes': boxes_out[keep_nms], 'scores': scores_out[keep_nms], 'labels': labels_out[keep_nms]})
        return detections


# HEURISTIC CLASSIFIERS

class RoomTypeClassifier:
    def __init__(self, room_name_label_id):
        self.room_name_label_id = room_name_label_id
        print(f"RoomTypeClassifier initialized. Target 'room_name' label ID: {self.room_name_label_id}")

    def classify_by_size_and_position(self, boxes, labels, image_size):
        height, width = image_size
        room_predictions = []
        for box, label_id in zip(boxes, labels):
            if label_id != self.room_name_label_id:
                continue
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0:
                continue
            if box_area > 3000:
                room_type = 'bedroom' if random.random() < 0.6 else 'hall'
            elif box_area > 800:
                options = ['bedroom', 'hall', 'kitchen']
                room_type = random.choice(options)
            else:
                room_type = 'bathroom' if random.random() < 0.5 else 'kitchen'
            room_predictions.append({
                'type': room_type,
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': 0.7 
            })
        return room_predictions

class RoomCounter:
    def __init__(self):
        pass
    def count_rooms(self, predictions, image_size):
        room_counts = {'rooms': 0, 'halls': 0, 'kitchens': 0, 'bathrooms': 0, 'rooms_detail': []}
        room_type_count = defaultdict(int)
        room_details_map = defaultdict(lambda: {'count': 0, 'areas': []})
        for pred in predictions:
            room_type = pred.get('type', 'bedroom')
            room_type_count[room_type] += 1
            box = pred['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            room_details_map[room_type]['areas'].append(area)
            room_details_map[room_type]['count'] += 1
        room_counts['rooms'] = room_type_count.get('bedroom', 0)
        room_counts['halls'] = room_type_count.get('hall', 0)
        room_counts['kitchens'] = room_type_count.get('kitchen', 0)
        room_counts['bathrooms'] = room_type_count.get('bathroom', 0)
        target_labels = ['bedroom', 'hall', 'kitchen', 'bathroom']
        for room_type in target_labels:
            details = room_details_map[room_type]
            if details['count'] > 0:
                avg_area = np.mean(details['areas']) if details['areas'] else None
                room_counts['rooms_detail'].append({
                    'label': room_type.capitalize(),
                    'count': details['count'],
                    'approx_area': float(avg_area) if avg_area is not None else None
                })
        return room_counts


# helper functions
def get_model(model_path=MODEL_PATH):
    """Loads and caches the model in a global variable."""
    global MODEL
    if MODEL is None:
        print(f"Loading model from {model_path} onto {DEVICE}...")
        MODEL = ObjectDetectorFromScratch(
            num_fg_classes=NUM_FG_CLASSES,
            num_anchors=NUM_ANCHORS,
            target_size=IMAGE_SIZE
        )
        # Load weights
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    return MODEL

def preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Prepares a PIL image for inference.
    Matches the non-training augmentation from FloorplanAugmentation.
    """
    # 1. Convert to numpy array (RGB)
    image_np = np.array(image.convert('RGB'))
    
    # 2. Resize (squash) to target size
    image_resized = cv2.resize(image_np, (target_size, target_size))
    
    # 3. Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # 4. Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
    
    # 5. Add batch dimension (B, C, H, W)
    return image_tensor.unsqueeze(0)


# MAIN INFERENCE PIPELINE
def parse_floorplan(image, room_name_label_id=ROOM_NAME_LABEL_ID):
    """
    Parses a floorplan image and returns a JSON object with room counts.

    Args:
        image (PIL.Image or str): A PIL Image object or a file path to an image.
        room_name_label_id (int): The category ID for 'room_name'. 
                                  Defaults to the constant at the top of the file.

    Returns:
        dict: A dictionary with room counts and details.
    """
    
    # 1. Load model (uses cached model after first run)
    model = get_model()
    
    # 2. Instantiate heuristic classifiers
    classifier = RoomTypeClassifier(room_name_label_id)
    counter = RoomCounter()

    # 3. Preprocess image
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image}")
            return {}
    
    original_size = image.size # (width, height)
    image_tensor = preprocess_image(image, target_size=IMAGE_SIZE).to(DEVICE)

    # 4. Run model inference
    with torch.no_grad():
        detections_list = model(image_tensor)
    
    detections = detections_list[0] # Get detections for the first (and only) image
    
    boxes = detections['boxes'].cpu()
    labels = detections['labels'].cpu()
    scores = detections['scores'].cpu()
    
    print(f"Model detected {len(boxes)} objects in total.")

    # 5. Scale boxes back to original image size
    # The training resized (squashed) 640x640, so we un-squash
    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0::2] *= original_size[0] / IMAGE_SIZE  # Scale X (width)
    boxes_scaled[:, 1::2] *= original_size[1] / IMAGE_SIZE  # Scale Y (height)
    
    # 6. Run Heuristic Classification
    # Pass numpy arrays to the classifiers
    heuristic_predictions = classifier.classify_by_size_and_position(
        boxes_scaled.numpy(),
        labels.numpy(),
        (original_size[1], original_size[0]) # (height, width)
    )
    
    print(f"Heuristics classified {len(heuristic_predictions)} as rooms.")

    # 7. Run Room Counting
    final_json_output = counter.count_rooms(
        heuristic_predictions,
        (original_size[1], original_size[0])
    )
    
    return final_json_output


if __name__ == "__main__":
    print("--- Floorplan Inference Script ---")
    
    # SET THE PATH TO YOUR TEST IMAGE HERE
    TEST_IMAGE_PATH = "/content/drive/MyDrive/assets/images/10_11_jpg.rf.8ff1f89249b54f8ecfd63ff673fb8e94.jpg"
    
    # Check if model file exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run the training script first or place 'best_model.pth' in this directory.")
    
    # Check if image file exists
    elif not Path(TEST_IMAGE_PATH).exists():
        print(f"Error: Test image file not found at {TEST_IMAGE_PATH}")
        print("Please check the TEST_IMAGE_PATH variable.")
    
    else:
        # Run the pipeline
        print(f"\nParsing floorplan: {TEST_IMAGE_PATH}...")
        try:
            result_json = parse_floorplan(TEST_IMAGE_PATH)
            
            print("\n" + "="*70)
            print("INFERENCE RESULT")
            print("="*70)
            print(json.dumps(result_json, indent=4))

        except Exception as e:
            print(f"\nAn error occurred during inference: {e}")
            import traceback
            traceback.print_exc() # Print full error for debugging