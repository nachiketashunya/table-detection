# Table Detection using CascadeTabNet

This repository implements an active learning pipeline for table detection using the **CascadeTabNet** model, leveraging **ICDAR19** and **Marmot** datasets. The research focuses on achieving state-of-the-art performance with minimal labeled data by iteratively training and retraining the model using confidence-based querying.

## Motivation

The core of this project is to reduce the amount of labeled data required to train robust table detection models by employing active learning techniques. By querying only the most informative samples from the dataset based on model confidence, we can significantly reduce labeling efforts while maintaining or improving model performance.

## Key Contributions

- **Active Learning Loop**: Implements an active learning strategy that selects samples for labeling based on confidence scores, minimizing the amount of training data required.
- **Minimal Data, Maximum Results**: Iteratively refines model performance by leveraging smaller fractions of the dataset in each cycle, retraining with undetected tables, and focusing on performance metrics.
- **Confidence-Based Querying**: Selects the most uncertain predictions for labeling to maximize the model’s learning efficiency.

## Process Overview

The pipeline follows this step-by-step process:

1. **Dataset Splitting**:
   - The dataset is split into training and testing sets.
   - The training set is further divided into a labeled subset (initial fraction) and an unlabeled pool.

2. **Initial Training**:
   - Train the CascadeTabNet model on a small fraction of the labeled data.

3. **Inference**:
   - Perform inference on the unlabeled dataset and evaluate the confidence scores of the predictions.

4. **Confidence-Based Selection**:
   - Identify the most uncertain predictions based on a predefined confidence threshold.

5. **Querying and Augmentation**:
   - Query labels for the most uncertain samples and augment the labeled training set with these newly labeled samples.

6. **Retraining**:
   - Retrain the model with the augmented training set.

7. **Performance Evaluation**:
   - Evaluate the model performance on the test set and report metrics like precision, recall, and F1-score.

8. **Iteration**:
   - Repeat steps 2-7 for different fractions of the dataset to achieve optimal results.

