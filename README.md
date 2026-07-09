#  HomeValue-AI

An end-to-end regression project that predicts residential home sale prices using the Ames Housing dataset (Kaggle), covering the complete machine learning workflow — from raw, messy data to a deployed, interactive prediction app.

**Author:** Amna Arshad — AI Engineer

---

##  Project Overview

Real estate pricing depends on dozens of interacting factors — size, quality, location, age, and condition. This project builds a regression pipeline that learns these relationships from historical sales data (1,460 homes, 79 raw features) and predicts prices for unseen properties, while explicitly comparing regularization techniques to handle high-dimensional, real-world data.

The project was built to demonstrate a complete, production-aware regression workflow — not just model fitting, but the reasoning behind every preprocessing decision, model comparison, and known limitation.

---

##  What This Project Demonstrates

- Handling **messy, real-world data** with context-aware missing value imputation
- Statistically-informed **outlier detection and treatment**
- Correcting **skewed distributions** via log transformation
- Proper **categorical encoding** strategy (ordinal vs. nominal)
- Preventing **data leakage** during feature scaling
- Comparing **Linear, Ridge, and Lasso Regression** to understand the bias-variance trade-off
- **Hyperparameter tuning** with cross-validation
- Recognizing and communicating a real model limitation (**extrapolation risk**)
- **Deploying** the trained model in an interactive Streamlit application

---

##  Dataset

**Source:** [Ames Housing Dataset — Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- 1,460 training records, 1,459 test records
- 79 raw features covering structural details, quality ratings, location, and sale conditions
- Target variable: `SalePrice`

---

##  Methodology

### 1. Data Cleaning & Missing Value Handling
Missing values were not filled blindly — each column was evaluated for *why* it was missing:
- Columns with 90%+ missing data (`PoolQC`, `MiscFeature`, `Alley`, `Fence`) were dropped
- Columns where `NaN` meant "feature doesn't exist" (e.g., no garage, no fireplace) were filled with `'None'` / `0`
- Genuinely missing values (e.g., `LotFrontage`) were imputed using a **grouped median** (by `Neighborhood`), since lot size is location-dependent
- Minor gaps in the test set were filled using mode/zero imputation

### 2. Outlier Detection
- Used the **IQR method** (chosen over Z-score, since the data is right-skewed)
- Statistically flagged outliers were visually inspected rather than removed automatically
- Only 2 genuine anomalies were removed (very large living area paired with an unrealistically low price) — legitimate high-value homes flagged by IQR were intentionally kept

### 3. Target Transformation
- `SalePrice` was right-skewed (skewness ≈ 1.88)
- Applied `log1p` transformation to normalize the distribution, improving linear model performance

### 4. Feature Encoding
- **Ordinal encoding** for quality/rating columns (`ExterQual`, `BsmtQual`, `KitchenQual`, etc.), manually mapped to preserve their natural ranking
- **One-Hot encoding** for nominal columns (`Neighborhood`, `Foundation`, etc.), with train and test sets combined before encoding to guarantee consistent columns
- Final feature set: **193 engineered features**

### 5. Feature Scaling
- `StandardScaler` fit **only on training data**, then applied to validation and test sets — avoiding data leakage from improperly re-fitting the scaler

### 6. Model Building & Comparison

| Model | Validation RMSE | Validation R² |
|---|---|---|
| Linear Regression | 0.1314 | 0.8975 |
| Ridge Regression | 0.1258 | 0.9061 |
| Lasso Regression | 0.1245 | 0.9081 |
| **Lasso (tuned, GridSearchCV)** | **0.1235** | **0.9096** |

- Hyperparameters tuned via **5-fold GridSearchCV**
- **Lasso Regression** was selected as the final model — it not only achieved the best validation performance but also performed automatic feature selection, eliminating 128 of 193 features as statistically unimportant

### 7. Model Evaluation
- Actual vs. Predicted plots confirmed a strong fit, particularly in the mid-to-high price range
- Slightly higher variance was observed in the low-price segment, attributable to fewer training examples in that range

### 8. Handling Extrapolation Risk
While generating final predictions, one test case produced an unrealistic price (~$1.7M) for a property with feature values (living area, basement size) far outside the training data's range. This surfaced an important limitation of linear models: **unreliable extrapolation beyond the training distribution.** Predictions were capped within a realistic bound derived from the training data, and this same safeguard was carried into the deployed app.

---

##  Deployment — Streamlit App

The final tuned Lasso model was deployed as an interactive web app where users can input house characteristics (size, quality, bedrooms, neighborhood, etc.) and receive an instant price estimate, along with a feature-importance breakdown showing which inputs are driving the prediction.

---

##  Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Modeling:** Scikit-learn (Linear, Ridge, Lasso Regression, GridSearchCV)
- **Deployment:** Streamlit
- **Model Persistence:** Joblib

---

