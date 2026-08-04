# 🩺 Diabetes Prediction using Gaussian Naive Bayes

An end-to-end Machine Learning project that utilizes health metrics to predict whether a patient has diabetes using the **Gaussian Naive Bayes** algorithm.

---

## 📌 Project Overview

This project explores a medical dataset containing patient demographics and health metrics (e.g., Blood Glucose, HbA1c, BMI, Age). The goal is to build a baseline classification model to identify diabetic patients, attempt model tuning techniques, and thoroughly evaluate its performance on imbalanced data.

---

## 🛠️ Workflow & Methodology

### 1. Exploratory Data Analysis (EDA) & Cleaning
* **Data Inspection:** Handled missing/uninformative values and removed minority outlier rows (e.g., `gender == 'Other'`).
* **Feature Analysis:** Analyzed feature distributions (histograms/boxplots) and checked correlation matrices. Identified key continuous features like `HbA1c_level` and `blood_glucose_level`.
* **Preprocessing:** Applied One-Hot Encoding (`pd.get_dummies`) to categorical features (`gender`) while preserving floating-point data types for numerical features.

### 2. Model Training & Optimization Experiments
* Split dataset into **80% Training** and **20% Testing** sets using stratified sampling (`stratify=y`) to preserve class proportions.
* Trained a baseline **Gaussian Naive Bayes (`GaussianNB`)** classifier.
* **Optimization Experiments Attempted:**
  1. **Probability Threshold Tuning:** Adjusted the decision threshold (`threshold = 0.4`) via `predict_proba` to improve recall.
  2. **Class Priors Adjustment:** Configured balanced priors (`priors=[0.5, 0.5]`) to force equal class importance during training.
  3. **Feature Selection:** Trained the model exclusively on top predictors (`blood_glucose_level`, `HbA1c_level`, `age`, `bmi`, `hypertension`).

---

## 📊 Results & Key Findings

### Baseline Model Performance
* **Overall Accuracy:** ~91%
* **Diabetes Class (Class 1) F1-Score:** ~0.54
* **Diabetes Class (Class 1) Recall:** ~0.65

### 🧪 Tuning Experiments Outcome
Despite testing multiple optimization strategies (threshold adjustments, equal priors, and selecting top continuous features), the overall **F1-score for the positive class (Diabetes) did not significantly improve**.

### ⚠️ Performance Analysis: Class Imbalance Challenge
Although overall accuracy appears high (~91%), **accuracy is a misleading metric for this dataset**. 

* **Why the model struggled:** The dataset is heavily imbalanced (~91.5% non-diabetic vs. ~8.5% diabetic). Because Gaussian Naive Bayes relies on strong conditional probability assumptions and independent feature distributions, standard optimization techniques failed to overcome the underlying class imbalance.
* **Key Takeaway:** For highly imbalanced medical classification tasks, Gaussian Naive Bayes serves as a solid baseline, but achieving higher precision and F1-score requires advanced techniques like resampling (e.g., SMOTE) or shifting to tree-based algorithms (e.g., Random Forest / XGBoost).

---

## 📁 Project Structure

```text
├── data/
│   ├── raw_data.csv             # Raw dataset
│   └── processed_data.csv       # Cleaned dataset
├── notebooks/
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
└── README.md