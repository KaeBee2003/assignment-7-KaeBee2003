# Multi-Class Classification: Model Evaluation & Comparison

A comprehensive machine learning project evaluating multiple classification algorithms on satellite image data, with detailed ROC and Precision-Recall curve analysis.

## Project Overview

This project implements and compares six different classification models on a multi-class satellite image dataset. The analysis includes baseline performance evaluation, ROC curve analysis, Precision-Recall curve analysis, and ensemble method experimentation.

## Objectives

- Train and evaluate multiple classification algorithms
- Perform ROC (Receiver Operating Characteristic) analysis for model selection
- Conduct Precision-Recall Curve (PRC) analysis
- Compare model performance using multiple metrics
- Recommend the best model based on comprehensive evaluation
- Experiment with ensemble methods (Bonus)

## Dataset

- **Training Set**: 4,435 samples with 36 features
- **Test Set**: 2,000 samples with 36 features
- **Format**: Space-separated values
- **Type**: Multi-class classification problem
- **Files**: `sat.trn` (training), `sat.tst` (testing)

## Libraries

```python
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
```

## Models Implemented

### Baseline Models
1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree**
3. **Dummy Classifier** (baseline)
4. **Logistic Regression**
5. **Naive Bayes**
6. **Support Vector Machine (SVM)**

### Ensemble Models
7. **Random Forest**
8. **XGBoost**

## Project Structure

```
.
├── Ass7.ipynb                 # Main Jupyter notebook
├── README.md                  # This file
├── sat.trn                    # Training dataset
├── sat.tst                    # Test dataset
└── requirements.txt           # Python dependencies
```

## Results Summary

### Part A: Baseline Performance

| Model | Accuracy | Weighted F1-Score |
|-------|----------|-------------------|
| **KNN** | **0.9045** | **0.9037** |
| SVM | 0.8955 | 0.8925 |
| Decision Tree | 0.8505 | 0.8509 |
| Logistic Regression | 0.8395 | 0.8296 |
| Naive Bayes | 0.7965 | 0.8036 |
| Dummy | 0.2305 | 0.0864 |

### Part B: ROC-AUC Analysis

| Model | Macro-Averaged AUC |
|-------|-------------------|
| **SVM** | **0.9852** |
| KNN | 0.9786 |
| Logistic Regression | 0.9757 |
| Naive Bayes | 0.9553 |
| Decision Tree | 0.9002 |
| Dummy | 0.5000 |

### Part C: Precision-Recall Analysis

| Model | Macro-Averaged AP |
|-------|-------------------|
| **KNN** | **0.9217** |
| SVM | 0.9177 |
| Logistic Regression | 0.8711 |
| Naive Bayes | 0.8105 |
| Decision Tree | 0.7366 |
| Dummy | 0.1667 |

### Ensemble Methods

| Model | Accuracy | Weighted F1 | Macro-AUC |
|-------|----------|-------------|-----------|
| **Random Forest** | **0.9115** | **0.9094** | **0.9901** |
| XGBoost | 0.9050 | 0.9030 | 0.9900 |

## Final Recommendation

### **Recommended Model: K-Nearest Neighbors (KNN)**

#### Justification:
- ✅ **Highest Weighted F1** (0.9037) → Best balance between precision and recall
- ✅ **Highest Macro-Averaged AP** (0.9217) → Superior precision-recall trade-off
- ✅ **Near-optimal ROC-AUC** (0.9786) → Excellent discriminative ability
- ✅ **Interpretability** → Easy to explain and visualize

KNN provides the most **balanced performance** across all metrics, making it the most reliable choice for this multi-class classification task, especially when maximizing both precision and recall simultaneously.

### Trade-offs Analysis

- **SVM**: Highest ROC-AUC (theoretical separability) but slightly lower practical performance
- **KNN**: Best practical thresholded performance (PRC, F1)
- **Ensemble Methods**: Highest overall scores but increased complexity

## Key Insights

1. **ROC-AUC vs Precision-Recall**: ROC-AUC evaluates ranking quality across all thresholds, while PRC focuses on positive class quality—more useful for imbalanced datasets.

2. **Ensemble Power**: Random Forest and XGBoost achieved the highest scores, demonstrating the effectiveness of ensemble methods in capturing complex patterns.

3. **Baseline Importance**: The Dummy classifier (AUC ≈ 0.5) serves as a crucial sanity check, confirming other models learn meaningful patterns.

4. **Model Failure**: A deliberately misconfigured model (shuffled labels) achieved AUC < 0.5, demonstrating worse-than-random performance.

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (separability measure)
- **Precision-Recall AP**: Average precision across all recall thresholds

## Assignment Parts

- **Part A**: Data preparation and baseline model training
- **Part B**: ROC curve analysis and AUC comparison
- **Part C**: Precision-Recall curve analysis
- **Part D**: Final model recommendation with justification
- **Brownie Task**: Ensemble methods and poor-performing model demonstration
