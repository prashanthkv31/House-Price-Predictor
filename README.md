# Boston House Price Prediction

A machine learning project that predicts median house prices (MEDV) in Boston
using the Boston Housing dataset.

## What this project does
- Handles missing values (median imputation)
- Detects and removes outliers using the IQR method
- Reduces skewness with log transformation (CRIM, DIS)
- Scales features with StandardScaler (fit on train only, to avoid data leakage)
- Trains and compares Linear, Ridge and Lasso regression
- Tunes Lasso (alpha = 0.01) and evaluates with RMSE and R²
- Saves the final model and scaler with joblib

## Tech stack
Python, Pandas, NumPy, scikit-learn, Google Colab
