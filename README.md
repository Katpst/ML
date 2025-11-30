Machine Learning Projects

This repository contains a collection of small machine-learning projects demonstrating core ML concepts such as regression, classification, model evaluation and regularization.
Currently, it includes two Jupyter notebooks: one focused on overfitting in regression and another on credit default classification.

---

## Projects

### **1. Overfitting & Regularized Regression**

**Topics covered:**

* Linear Regression
* Overfitting & model complexity
* Lasso Regression (L1)
* Ridge Regression (L2)
* Train/test split
* Bias–variance trade-off
* Evaluation metrics:

  * RMSE
  * MAE
  * R²

**Description:**
This notebook shows how overfitting occurs in regression models and how regularization (Lasso and Ridge) can reduce complexity and improve generalization. It includes visualizations of coefficient shrinkage and performance across models.

---

### **2. Credit Default Classification (Lasso & KNN)**

**Notebook:** `credit_default_classification.ipynb`

**Topics covered:**

* Logistic Regression 
* K-Nearest Neighbors (KNN)
* Feature scaling (StandardScaler)
* Model comparison
* Evaluation metrics:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
  * ROC Curve & AUC

**Description:**
This notebook builds a simple classifier predicting whether a borrower will default. It compares logistic regression with Lasso regularization against KNN and highlights the most informative features.

---

**Main libraries:**

* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

---

Clone the repository:

```
git clone https://github.com/Katpst/ML
cd ML
```

## 📌 Future Work

* Add more classification models (SVM, Random Forest, XGBoost)
* Hyperparameter tuning (GridSearch, RandomSearch, Optuna)
* Add Elastic Net for regression
* Additional datasets (time series, NLP, etc.)



