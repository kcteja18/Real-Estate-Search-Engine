# Floorplan Room Count Detection

This project uses a Convolutional Neural Network (CNN) to analyze floorplan images and predict the count of different room types.

The model (`RoomDetectionCNN`) is built on a ResNet-18 backbone. Instead of detecting bounding boxes, it treats the task as a multi-head classification problem. Each room type (e.g., "bedroom," "kitchen") has a dedicated output head that classifies the count (e.g., 0, 1, 2... rooms) from a predefined range.

## 📊 Dataset & Split
The model was trained on a custom dataset of 545 floorplan images, with annotations provided in a COCO JSON file.

- **Data Source:** `annotations2.coco.json` and an associated image directory.
- **Labeling:** Room labels (bedroom, kitchen, etc.) are derived from the area of bounding box annotations.

**Dataset Split (80/10/10):**
- Training: 436 images (80%)
- Validation: 54 images (10%)
- Test: 55 images (10%)

## ⚙️ Training Hyperparameters
- Epochs: 50
- Batch Size: 32
- Learning Rate: 1.0e-3
- Optimizer: Adam
- Weight Decay: 1.0e-4
- Loss Function: `nn.CrossEntropyLoss` (applied to each prediction head)

## 📈 Evaluation Metrics (on Test Set)
The model's performance was evaluated on the 55-image test set.

- **Overall Room Count Accuracy:** 40.00%
	- This metric represents the percentage of test images where the model predicted the exact total number of rooms (e.g., predicted 6 total rooms, and the ground truth was 6).
- **Mean Absolute Error (MAE):** 2.3636 rooms
	- The average absolute difference between the predicted total room count and the true total room count.
- **IoU Approximation (Count-Based):** 0.5833
	- This is a count-based IoU, not a pixel or bounding-box IoU. It is calculated by summing the minimum counts for each room type (intersection) and dividing by the sum of the maximum counts (union).

### Per-Type Count Accuracy
This measures the accuracy of predicting the exact count for a specific room type.

- Bedroom: 70.91%
- Kitchen: 40.00%
- Bathroom: 32.73%
- Garage: 40.00%
- Hall: 50.91%
