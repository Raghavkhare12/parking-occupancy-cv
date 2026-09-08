# Project Statement

## 1. Project Title

**Computer Vision Based Parking Lot Occupancy Detection and Analysis System**

---

## 2. Problem Statement

Finding available parking spaces in a parking lot can be difficult when the number of vehicles and parking spaces is large. Manual monitoring of parking occupancy is time-consuming and does not provide a consistent method for estimating parking availability.

This project proposes a computer vision-based system for analyzing parking-lot images and determining whether individual parking spaces are empty or occupied.

---

## 3. Proposed Solution

The system processes parking-lot images using computer vision techniques. Individual parking-space regions are extracted using parking-space annotations. Each region is resized, converted to grayscale, normalized, and represented using Histogram of Oriented Gradients (HOG) features.

An SVM classifier is then used to classify each parking space as either empty or occupied.

The individual predictions are aggregated to calculate the overall occupancy of the parking lot.

---

## 4. Objectives

The project aims to:

1. Process parking-lot images using computer vision.
2. Extract individual parking-space regions.
3. Apply image preprocessing techniques.
4. Extract HOG features.
5. Train an SVM-based occupancy classifier.
6. Evaluate classification performance.
7. Analyze complete parking-lot images.
8. Generate occupancy statistics and reports.

---

## 5. Functional Requirements

### FR1 — Dataset Processing

The system shall read COCO-format parking-space annotations and generate parking-space image crops.

### FR2 — Image Preprocessing

The system shall resize, grayscale, and normalize parking-space images.

### FR3 — Feature Extraction

The system shall extract HOG features from preprocessed parking-space images.

### FR4 — Occupancy Classification

The system shall classify each parking space as empty or occupied.

### FR5 — Parking Analysis

The system shall calculate the total number of parking spaces, occupied spaces, and empty spaces.

### FR6 — Occupancy Calculation

The system shall calculate the percentage occupancy of the parking lot.

### FR7 — Visualization

The system shall generate an annotated parking-lot image containing parking-space predictions.

### FR8 — Reporting

The system shall generate CSV reports containing individual parking-space predictions and overall occupancy statistics.

---

## 6. Non-Functional Requirements

### NFR1 — Accuracy

The system should provide reliable classification performance on unseen parking-space images.

### NFR2 — Maintainability

The implementation should be modular, with separate modules for dataset processing, preprocessing, classification, analysis, evaluation, and reporting.

### NFR3 — Reliability

The system should validate input files and handle invalid or missing input data appropriately.

### NFR4 — Reproducibility

The project should provide a reproducible training and evaluation workflow.

### NFR5 — Usability

The system should be executable through a simple command-line interface.

### NFR6 — Testability

Core functionality should be covered by automated tests.

---

## 7. Expected Inputs

The system accepts:

- Parking-lot image
- COCO-format parking-space annotation file
- Trained occupancy classification model

---

## 8. Expected Outputs

The system produces:

- Annotated parking-lot image
- Individual parking-space CSV report
- Overall occupancy summary
- Model evaluation metrics
- Confusion matrices

---

## 9. Machine Learning Method

The system follows the workflow below:

```text
Parking-Lot Image
        ↓
Parking-Space Regions
        ↓
64 × 64 Image Preprocessing
        ↓
Grayscale + Normalization
        ↓
HOG Feature Extraction
        ↓
RBF SVM Classifier
        ↓
Empty / Occupied
        ↓
Occupancy Analysis
        ↓
Reports and Visualization
````

### HOG Configuration

The HOG feature extraction uses:

* Orientations: `9`
* Pixels per cell: `(8, 8)`
* Cells per block: `(2, 2)`
* Block normalization: `L2-Hys`

The resulting HOG feature vector contains **1,764 features**.

### SVM Configuration

The occupancy classifier uses an RBF-kernel Support Vector Machine with:

* Kernel: `RBF`
* `C`: `10.0`
* Gamma: `scale`
* Probability calibration: `CalibratedClassifierCV`
* Calibration method: `sigmoid`
* Cross-validation: `5-fold`

---

## 10. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Validation Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 96.05% |
| Precision | 93.90% |
| Recall    | 98.50% |
| F1 Score  | 96.14% |

### Final Test Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **96.50%** |
| Precision | **95.32%** |
| Recall    | **97.80%** |
| F1 Score  | **96.54%** |

The final test set contains **2,000 samples**, consisting of:

* 1,000 empty parking-space images
* 1,000 occupied parking-space images

---

## 11. Sample Application Result

The system was tested on a sample parking-lot image containing **100 parking spaces**.

The resulting analysis was:

| Measure              | Result |
| -------------------- | -----: |
| Total parking spaces |    100 |
| Occupied spaces      |     59 |
| Empty spaces         |     41 |
| Occupancy percentage | 59.00% |

The system also generates an annotated image showing the predicted status of individual parking spaces.

---

## 12. Project Scope

The current implementation focuses on offline parking-lot image analysis.

Parking-space locations are obtained from provided parking-space annotations. The machine-learning component performs occupancy classification for those spaces.

The following are outside the current scope:

* Real-time video processing
* Automatic parking-space discovery
* Web interfaces
* Database integration
* Mobile application development

---

## 13. Deliverables

The project includes:

* Python source code
* Dataset preparation module
* Image preprocessing module
* HOG feature extraction
* SVM classifier
* Model evaluation module
* Parking-lot analyzer
* Reporting module
* Automated tests
* README documentation
* Project statement
* Design and project report artifacts

