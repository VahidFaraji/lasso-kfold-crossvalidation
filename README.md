# 📈 LASSO Regression with K-Fold Cross Validation

This project demonstrates how to use **K-Fold Cross Validation** to tune the regularization parameter **λ** for the **LASSO** regression model. It evaluates model performance via RMSE, and includes both signal reconstruction and audio denoising components.

---

## 🧠 Topics Covered

- LASSO via coordinate descent
- Cross-validation for λ selection
- RMSE plots for training vs validation
- Sparse signal recovery
- Denoising audio using spectral sparsity

---

## 📁 Folder Structure

```text
lasso-kfold-crossvalidation/
├── code/
│   ├── lasso.py            # LASSO, CV, multiframe, denoise functions
│   └── tasks.py            # CLI-based runner for tasks 4–7
│   └── Task4_5.ipynb       # Jupyter notebook for LASSO analysis
├── data/
│   └── A1_data.mat         # Input data (X, t, audio, etc.)
├── Report/                # Will hold plots and denoised audio (optional)
├── README.md
