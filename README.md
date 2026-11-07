# Chest CT Classification Using Transfer Learning and Attention-Based MIL

An experimental deep learning project for three-class chest CT image classification using PyTorch, transfer learning, and attention-based Multiple Instance Learning (MIL).

## Overview

This project explores two approaches to classifying chest CT images into:

- COVID
- Healthy
- Others

The first approach uses a pretrained ResNet-18 model to classify individual CT slices, followed by mean-probability aggregation to generate patient-level predictions.

The second approach uses an attention-based Multiple Instance Learning architecture. It treats all slices belonging to a patient as a single bag and learns attention weights to aggregate slice-level features into a patient-level prediction.

## Dataset

**Soares and Angelov SARS-CoV-2 CT Scan Dataset**

Source: https://doi.org/10.7910/DVN/SZDUQX

Dataset summary:

- 4,171 PNG images originally available.
- 2 exact duplicate pairs excluded, leaving 4,169 images.
- 210 patient folders across three classes.

| Class | Patient folders | Images |
|---|---:|---:|
| COVID | 80 | 2,167 |
| Healthy | 50 | 757 |
| Others | 80 | 1,247 |
| **Total** | **210** | **4,171** |

## Methodology

### 1. Data auditing and preprocessing

- Inspected image dimensions and class distribution.
- Validated PNG files and checked for corrupted images.
- Identified exact duplicate images using hashing.
- Excluded one image from each duplicate pair.
- Resized images to 224 × 224.
- Applied ImageNet normalization.
- Used a patient-folder-level split to avoid distributing slices from the same folder across different splits.

### 2. Dataset split

| Split | Patient folders |
|---|---:|
| Training | 147 |
| Validation | 31 |
| Test | 32 |
| **Total** | **210** |

### 3. ResNet-18 baseline

- Used pretrained ResNet-18 weights.
- Initially froze the convolutional backbone and trained the classification layer.
- Fine-tuned the final residual block and classifier.
- Used class-weighted cross-entropy loss.
- Aggregated slice-level softmax probabilities by taking their mean for each patient.

### 4. Attention-Based Multiple Instance Learning

- Extracted 512-dimensional slice features using the frozen Phase 2 ResNet-18 feature extractor.
- Grouped slice features into patient-level bags.
- Implemented a trainable attention mechanism to assign weights to individual slices.
- Used weighted feature aggregation to create a patient representation.
- Trained a three-class patient-level classifier using class-weighted cross-entropy loss.
- Selected the best checkpoint using validation macro-F1 and early stopping.

## Results

Both models were evaluated on the same 32 held-out patient folders.

| Metric | ResNet-18 | Attention MIL |
|---|---:|---:|
| Accuracy | 78.13% | 78.13% |
| Macro-F1 | 0.7747 | 0.7739 |

### Class-wise F1 scores

| Class | ResNet-18 | Attention MIL |
|---|---:|---:|
| COVID | 0.82 | 0.83 |
| Healthy | 0.71 | 0.71 |
| Others | 0.80 | 0.78 |

The models achieved comparable overall performance, with different class-wise error patterns.

## Interpretability

The attention-based MIL model produces a weight for each slice in a patient's CT scan.

These weights were extracted and visualised to inspect which slices contributed more strongly to the model's patient-level representation.

Attention weights represent model-derived importance scores and should not be interpreted as clinical explanations or lesion localisation.

## Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Limitations

- The held-out test set contains only 32 patient folders.
- Patient identity is represented using class-qualified folder names; the dataset's original patient-ID convention requires further verification.
- MIL uses features from a ResNet-18 trained with slice-level supervision.
- The models may be affected by class imbalance, label noise and overfitting.
- Attention weights do not establish medical relevance.
- The earlier recorded ResNet-18 evaluation differs from the reconstructed evaluation and remains under investigation.
- This project is experimental and is not intended for clinical diagnosis.

## Future Work

- Verify patient identity conventions and dataset metadata.
- Investigate the original baseline evaluation discrepancy.
- Explore alternative feature extractors and MIL architectures.
- Evaluate robustness using repeated patient-grouped cross-validation.
- Improve interpretability and error analysis.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, screen or make treatment decisions.
