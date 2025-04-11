# 📈 LASSO Regression with K-Fold Cross Validation

This project demonstrates how to apply **K-Fold Cross Validation** to tune the regularization parameter **λ** for the **LASSO** regression model. The goal is to select the λ that minimizes validation error and avoids overfitting or underfitting.

---

## 🔍 Problem Overview

We simulate a sparse linear regression problem and evaluate how different λ values affect the model's performance. Using **K-Fold Cross Validation**, we systematically:

- Split the data into `K` folds
- Train and validate LASSO models for different λ
- Select the λ with lowest average RMSE

---

## 🧪 Methods

- LASSO implementation via **Coordinate Descent**
- `K = 5` folds cross-validation
- RMSE evaluation across validation sets

---

## 📁 Files

- `code/lasso_cv.py` – Cross-validation driver
- `code/lasso_ccd.py` – LASSO implementation using coordinate descent
- `results/rmse_vs_lambda.png` – RMSE curve across λ values
- `results/optimal_lambda_result.png` – Prediction with best λ

---

## 📊 Example Output

<p float="left">
  <img src="results/rmse_vs_lambda.png" width="400"/>
  <img src="results/optimal_lambda_result.png" width="400"/>
</p>

---

## 🧠 Author

**Vahid Faraji**  
Master’s Student – Machine Learning, Systems and Control  
Lund University, Sweden
