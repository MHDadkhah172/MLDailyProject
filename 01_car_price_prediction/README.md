# 🚗 Project 01: Car Price Prediction

## 📌 Overview
This directory contains the inaugural project of the daily machine learning initiative. The core objective is to establish a foundational data processing workflow and apply predictive logic to estimate automobile prices. Beyond simple implementation, this project serves as a deep dive into the mathematical paradoxes of regression models, specifically examining how algorithms interpret non-linear, real-world dynamics like exponential depreciation.

## 📊 Dataset
The raw data is maintained within the `data/` directory (`car data.csv`). It includes essential features utilized to train the model and validate our theoretical assumptions regarding market behavior.

## ⚙️ Algorithmic Focus & System Logic
The implementation progressed through several critical analytical stages to ensure algorithmic stability:

1. **Data Ingestion & Inspection:** Loading the dataset and evaluating structural integrity, handling binary categorical mapping via Pandas.
2. **Resolving Multicollinearity:** Identified the algorithmic confusion (weight tug-of-war) caused by polynomial features (e.g., `Age_Squared`). Restructured the mathematical approach to prevent conflicting feature weights.
3. **Log-Log Architecture Implementation:** Modeled the real-world exponential decay of vehicle value by applying logarithmic transformations (`np.log` and `np.log1p`) to both the target variable (`Selling_Price`) and large-scale features (`Present_Price`, `Kms_Driven`). This resolved the "Linear Disconnect" paradox and prevented computational explosion.
4. **Algorithmic Scaling:** Applied `MinMaxScaler` to isolate feature variance securely within a `[0, 1]` boundary, preserving the physical logic of the data (avoiding negative vehicle ages).
5. **Systemic Validation (K-Fold):** Replaced standard train-test splitting with **K-Fold Cross-Validation** to expose and eliminate sampling bias. The final model's reliability is measured by transforming logarithmic predictions back to real-world currency scales via exponential functions (`np.exp`) to calculate a strictly accurate Mean Absolute Error (MAE).