# 💳 Personal Loan Acceptance Prediction

This project focuses on predicting whether a bank customer will accept a personal loan offer. It covers end-to-end machine learning workflows including Exploratory Data Analysis (EDA), model training, imbalanced data handling, and decision threshold optimization.

---

### 1. Exploratory Data Analysis (EDA) & Data Preprocessing
* **Dataset Cleaning:** Dropped non-informative identifiers (`ID`, `ZIP Code`) and rectified invalid entries (e.g., converting negative values in `Experience` to non-negative equivalents).
* **Statistical & Feature Analysis:** Analyzed value distributions, checked for missing data, and examined feature correlations (e.g., income vs. credit card spend) to understand key predictors of loan acceptance.
* **Feature Scaling:** Applied `StandardScaler` to zero-mean and unit-variance normalize continuous numerical features, ensuring uniform distance metrics for KNN and faster convergence for Logistic Regression.
* **Stratified Data Splitting:** Executed `train_test_split` with `test_size=0.2`, `random_state=0`, and `stratify=y` to preserve the $9:1$ imbalanced class ratio of loan acceptance across both training ($4000$ samples) and testing ($1000$ samples) sets.
---

### 2. Model Development & Experiments

#### **A. Logistic Regression**
* Evaluated performance across multiple solvers (`liblinear`, `lbfgs`, `newton-cg`, `sag`, `saga`).
* Integrated `class_weight='balanced'` to handle the inherent class imbalance in loan acceptances.

#### **B. K-Nearest Neighbors (KNN)**
* **Optimal $K$ Selection:** Plotted accuracy curves across different $K$ values to determine $K=5$ as the baseline parameter.
* **Weights Exploration:** Evaluated both `uniform` and `distance` weighting schemes to assess decision boundary behavior.

---

### 3. Handling Class Imbalance & Optimization

To address the minority target class and optimize performance:

* **SMOTE Sampling:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to balance the training set, substantially increasing model Recall (from $\approx 55\%$ to $>85\%$).
* **Decision Threshold Tuning:** Conducted fine-grained threshold grid searches (testing values from $0.1$ to $0.9$) directly on model predicted probabilities (`predict_proba`).
* **Optimal Trade-off:** Identified **$0.21$** as the optimal threshold, achieving a superior **F1-Score ($\approx 0.7528$)** while maintaining high Precision without artificial oversampling noise.

---

## 📊 Key Results Summary

* **KNN Baseline ($K=5$, Threshold $0.5$):** Precision-focused but suffered from low Recall ($\approx 55\%$).
* **KNN + Custom Threshold ($0.21$):** Achieved balanced classification performance with an **F1-Score of $0.7528$**, $95.60\%$ Accuracy, and $81.71\%$ Precision.