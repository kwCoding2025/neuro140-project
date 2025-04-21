Below is a revised Product Requirements Document (PRD) tailored to your specifications. It incorporates the answers to the clarification questions, removes personal names and references to xAI, and provides deeply technical instructions aligned with the research question and proposed extensions from the Midterm Report. The example JSON provided is used to define the expected output format, prioritizing higher-level features (rooms, walls), and the focus is solely on analysis without image generation.

---

# Product Requirements Document (PRD)

## Project Title
FloorPlanCAD Transformer Analysis System

## Version
1.0

## Date
March 25, 2025

---

## 1. Purpose and Scope

### 1.1 Objective
The system evaluates the capability of transformer-based image classification models to ingest architectural floorplans from PKL files in `./floorplancad-processed/pkl/train-00` and `train-01` (image-JSON pairs) and produce JSON representations capturing building features, focusing on higher-level entities like rooms and walls. It compares a fine-tuned Vision Transformer (ViT) against a ResNet50-based CNN baseline to assess improvements in feature identification and structural consistency.

### 1.2 Scope
- **Input**: PKL files from `./floorplancad-processed/pkl/train-00` and `train-01`, containing PNG images and JSON annotations (example format provided).
- **Output**: JSON representations detailing rooms (e.g., dimensions, boundaries) and walls (e.g., paths, coordinates), mirroring the provided example structure.
- **Models**: Fine-tuned ViT, ResNet50 CNN benchmark, with optional extensions to CADTransformer and Mask2Former.
- **Evaluation**: Composite loss prioritizing room count and wall consistency, with an optional IoU metric.
- **Constraints**: Limited to specified directories; no additional data assumed.

---

## 2. Functional Requirements

### 2.1 Data Ingestion
- **FR1**: Load PKL files from `./floorplancad-processed/pkl/train-00` and `train-01` using `pickle` in Python, extracting PNG images and JSON annotations.
  - Validate each pair: ensure PNG is readable via `PIL.Image.open()` and JSON adheres to the example schema (e.g., contains `width`, `height`, `layers`).
- **FR2**: Implement a 70/15/15 train/validation/test split using `sklearn.model_selection.train_test_split` with a fixed seed (e.g., 42) for reproducibility.
  - Output: Three datasets saved as PKL files in `./split/train.pkl`, `./split/val.pkl`, `./split/test.pkl`.

### 2.2 Data Preprocessing
- **FR3**: Apply augmentation to training images using `torchvision.transforms`:
  - Horizontal/vertical flips (`RandomHorizontalFlip`, `RandomVerticalFlip`, p=0.5).
  - Rotations (`RandomRotation`, degrees=(-90, 90)).
  - Scaling (`RandomResizedCrop`, scale=(0.8, 1.2), size=224).
  - Normalize to ViT/ResNet50 input (224x224, RGB, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- **FR4**: Convert JSON annotations to a tensor representation:
  - Extract `layers` paths (e.g., `d` field) and `points` as sequences of [x, y] coordinates.
  - Map `semantic-id` and `instance-id` to feature labels (e.g., "17" for walls, based on example).
  - Normalize coordinates by dividing by `width` and `height`.

### 2.3 Model Architecture
- **FR5**: Fine-tune a ViT model (`vit_base_patch16_224` from Hugging Face `transformers`):
  - Freeze all layers except the last two transformer blocks and the classification head.
  - Replace head with a custom MLP predicting feature counts (e.g., rooms, walls) and bounding box coordinates.
  - Input: 224x224 RGB images; Output: [num_rooms, num_walls, wall_coords].
- **FR6**: Implement a ResNet50 CNN baseline (`torchvision.models.resnet50`):
  - Pretrained on ImageNet, replace final FC layer with a head matching ViT output.
  - Input: 224x224 RGB images; Output: [num_rooms, num_walls, wall_coords].
- **FR7** (Optional): Integrate CADTransformer (pretrained weights from GitHub, if available):
  - Tokenize graphical primitives (e.g., lines from `d`) using its attention mechanism.
  - Fine-tune on the dataset for panoptic symbol spotting.
- **FR8** (Optional): Implement Mask2Former (from `transformers`):
  - Treat paths as 2D point sets; predict masks for rooms and walls.
  - Requires converting `points` to a point cloud representation.

### 2.4 Training Pipeline
- **FR9**: Use PyTorch for training:
  - Optimizer: AdamW (lr=1e-4, weight_decay=0.01).
  - Scheduler: CosineAnnealingLR (T_max=50 epochs).
  - Batch size: 32 (adjustable based on GPU memory).
  - Augmentations applied on-the-fly via a custom `Dataset` class.
- **FR10**: Define a composite loss function prioritizing rooms and walls:
  - Loss = w1 * MSE(num_rooms_pred, num_rooms_true) + w2 * MSE(num_walls_pred, num_walls_true) + w3 * L1(wall_coords_pred, wall_coords_true).
  - Weights: w1=0.5, w2=0.4, w3=0.1 (tunable via grid search).
  - Compute `num_rooms` by clustering wall intersections; `num_walls` from unique `semantic-id` "17" instances.
- **FR11**: Log metrics (train/eval loss, feature counts) using `tensorboard`; save checkpoints every 5 epochs.

### 2.5 JSON Decoding
- **FR12**: Convert model outputs to JSON matching the example schema:
  - `width`, `height`: Fixed at 100.0 (normalized input assumption).
  - `layers`: List of dictionaries with:
    - `d`: SVG path string reconstructed from predicted `wall_coords` (e.g., "M x1,y1 L x2,y2").
    - `stroke`: Assign "rgb(0,178,0)" for walls (semantic-id "17"), "rgb(178,0,178)" for others.
    - `points`: Interpolated coordinates between predicted endpoints.
    - `semantic-id`, `instance-id`: Infer from clustering (e.g., "17" for walls, "-1" if unclassified).
  - Handle edge cases: Merge overlapping walls (distance < 0.01 normalized units); ignore incomplete rooms.
- **FR13**: Validate output JSON against ground truth for structural consistency.

### 2.6 Evaluation Metrics
- **FR14**: Compute composite loss on test set:
  - Room count accuracy: |pred - true| / true.
  - Wall count accuracy: |pred - true| / true.
  - Wall coordinate error: Mean L1 distance between predicted and true `points`.
  - Aggregate: Weighted sum (room=0.5, wall_count=0.3, wall_coords=0.2).
- **FR15** (Optional): Implement IoU metric:
  - Split images into overlapping 112x112 patches; predict separately.
  - Compute IoU for wall masks at overlaps using a helper function to align coordinates.
  - Requires rasterizing `points` to binary masks (e.g., via `skimage.draw`).

### 2.7 Experiments
- **FR16**: Run two primary experiments:
  - **Baseline**: ResNet50 CNN, 50 epochs, composite loss.
  - **ViT**: Fine-tuned ViT, 50 epochs, composite loss.
- **FR17** (Optional): Run extensions:
  - **CADTransformer**: Fine-tune for 20 epochs, compare panoptic quality.
  - **Mask2Former**: Train for 50 epochs, evaluate mask accuracy.
- **FR18**: Analyze results:
  - Plot loss curves, feature count errors via `matplotlib`.
  - Inspect misclassified samples (e.g., missing walls) manually.

---

## 3. Non-Functional Requirements

### 3.1 Performance
- **NFR1**: Process a batch of 32 images in <5 minutes on a single GPU (e.g., NVIDIA V100).
- **NFR2**: Complete training in <24 hours per model on the dataset.

### 3.2 Scalability
- **NFR3**: Handle additional PKL files in the same directory structure via a configurable path parameter.

### 3.3 Reliability
- **NFR4**: Gracefully handle corrupt PKL files (e.g., skip with logging via `logging` module).

### 3.4 Usability
- **NFR5**: Provide detailed logs (`logging.INFO`) and visualizations (TensorBoard, PNG plots) for debugging.

---

## 4. Assumptions and Constraints

### 4.1 Assumptions
- PKL files contain PNG images and JSON annotations in the example format.
- GPU with >12GB VRAM available for training.
- Pretrained ViT weights accessible via Hugging Face.

### 4.2 Constraints
- Limited to `train-00` and `train-01`; no validation/test directories assumed.
- Optional extensions (CADTransformer, Mask2Former, IoU) are time-permitting.
- Analysis-only focus; no image generation.

---

## 5. Deliverables
- **D1**: Python scripts:
  - `data_loader.py`: Ingestion and splitting.
  - `preprocess.py`: Augmentation and tensor conversion.
  - `models.py`: ViT, ResNet50, optional extensions.
  - `train.py`: Training pipeline with loss.
  - `decode.py`: JSON output generation.
  - `evaluate.py`: Metrics and visualizations.
- **D2**: Model weights (`.pth` files) for ViT and ResNet50.
- **D3**: Evaluation report (JSON) with composite loss per experiment.
- **D4**: Visualizations (TensorBoard logs, PNG plots of predicted vs. true).

---

## 6. Timeline
- **Week 1**: Data ingestion, splitting, preprocessing (FR1-FR4).
- **Week 2**: Model setup, training pipeline (FR5-FR11).
- **Week 3**: JSON decoding, composite loss evaluation (FR12-FR14).
- **Week 4**: Experiments, optional IoU (FR16-FR18, FR15).
- **Week 5**: Finalize deliverables (D1-D4).

---

## 7. Risks and Mitigation
- **Risk 1**: Inaccurate wall/room detection due to complex paths.
  - **Mitigation**: Use clustering (e.g., DBSCAN) on `points` to refine predictions.
- **Risk 2**: Composite loss misweights features.
  - **Mitigation**: Tune weights via grid search on validation set.
- **Risk 3**: GPU memory overflow.
  - **Mitigation**: Reduce batch size or gradient accumulation.

---

## 8. Success Criteria
- ViT outperforms ResNet50 by >5% in composite loss on test set.
- JSON outputs achieve >70% room count accuracy and >60% wall count accuracy.
- System processes all data without crashes.

---

This PRD provides a technical blueprint for implementing the system, focusing on the research question of transformer efficacy in floorplan analysis, with detailed instructions for each component. Let me know if further adjustments are needed!