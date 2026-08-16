# 🫀 Heart Disease Classification — ML Assignment 2

> **BITS Pilani | NSP4 | Machine Learning Assignment 2**

---

## a. Problem Statement

Heart disease is one of the leading causes of death worldwide. Early and accurate prediction of heart disease based on clinical parameters can significantly improve patient outcomes. This project implements and compares **five machine learning classification models** on the UCI Heart Disease dataset to predict the presence or absence of heart disease in patients.

The goal is to evaluate which algorithm performs best on this clinical dataset using multiple evaluation metrics, and deploy an interactive web application for real-time prediction and model comparison.

---

## b. Dataset Description

**Dataset:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
**Source:** UCI Machine Learning Repository (merged Cleveland + VA + Switzerland + Hungary)

| Property | Value |
|---|---|
| Total Instances | 920 rows |
| Features | 13 input features |
| Target | Binary (0 = No Disease, 1 = Disease) |
| Task | Binary Classification |
| Missing Values | Handled via median imputation |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age in years |
| sex | Categorical | Sex (1 = male, 0 = female) |
| cp | Categorical | Chest pain type (0–3) |
| trestbps | Numeric | Resting blood pressure (mm Hg) |
| chol | Numeric | Serum cholesterol (mg/dl) |
| fbs | Categorical | Fasting blood sugar > 120 mg/dl |
| restecg | Categorical | Resting ECG results |
| thalach | Numeric | Maximum heart rate achieved |
| exang | Categorical | Exercise-induced angina |
| oldpeak | Numeric | ST depression induced by exercise |
| slope | Categorical | Slope of peak exercise ST segment |
| ca | Numeric | Number of major vessels colored (0–3) |
| thal | Categorical | Thalassemia (3=normal, 6=fixed, 7=reversible) |
| **target** | Binary | **0 = No Disease, 1 = Disease** |

**Preprocessing:**
- Missing values (`?`) replaced with column median
- Target binarised: values > 0 mapped to 1 (disease)
- Features standardised using `StandardScaler` (fit on train, applied to test)
- 80/20 stratified train/test split (random_state=42)

---

## c. GitHub Repository Link

🔗 **[github.com/akashdeepjaiswal/ml-classification-app](https://github.com/akashdeepjaiswal/ml-classification-app)**

**Repository structure:**
```
ml-classification-app/
├── app.py                  ← Streamlit web application
├── train_models.py         ← Model training & evaluation script
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── test_data.csv           ← Test split (20% of dataset)
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── metrics_results.csv
└── data/
    └── heart_disease.csv   ← Full dataset (downloaded from UCI)
```

---

## d. Models Used & Evaluation Metrics

### 📊 Metrics Comparison Table

All models trained on the same 80% training split; evaluated on the same 20% test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8152 | 0.8912 | 0.8091 | 0.8725 | 0.8396 | 0.6249 |
| Decision Tree | 0.7554 | 0.7519 | 0.7767 | 0.7843 | 0.7805 | 0.5045 |
| kNN | **0.8696** | 0.9021 | **0.8611** | **0.9118** | **0.8857** | **0.7357** |
| Naive Bayes | 0.8261 | 0.8813 | 0.8365 | 0.8529 | 0.8447 | 0.6473 |
| Random Forest (Ensemble) | 0.8315 | **0.9029** | 0.8198 | 0.8922 | 0.8545 | 0.6586 |

> Dataset: 918 usable rows (after preprocessing) · 80/20 stratified split · StandardScaler applied

---

### 💡 Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved accuracy 0.8152 and AUC 0.8912. Strong linear baseline. Assumes linear decision boundary which slightly limits performance on this non-linear medical dataset. Coefficient analysis shows `cp`, `ca`, and `thal` as most influential features. |
| Decision Tree | Lowest accuracy (0.7554) and AUC (0.7519) among all models. Without depth pruning it overfits the training data, leading to the weakest generalisation. Highly interpretable but not recommended for deployment on this dataset. |
| kNN (k=7) | **Best overall** — highest accuracy (0.8696), F1 (0.8857), Precision (0.8611), Recall (0.9118), and MCC (0.7357). The dataset's moderate size and 13 well-scaled features suit a distance-based model. k=7 provides good bias-variance balance. |
| Naive Bayes | Accuracy 0.8261 with strong recall (0.8529). The Gaussian independence assumption is imperfect for clinical data but still generalises well. Best choice when speed and simplicity are priorities. Useful for initial screening. |
| Random Forest (Ensemble) | Accuracy 0.8315 with highest AUC (0.9029). Ensemble of 200 trees gives robust discrimination ability. While kNN edges it on F1, Random Forest's superior AUC makes it the best model for ranked risk scoring. |
| **Overall Winner** | **kNN (k=7)** — highest F1 score (0.8857) and MCC (0.7357), with the best precision-recall balance. Runner-up: **Random Forest** (best AUC = 0.9029). |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/akashdeepjaiswal/ml-classification-app.git
cd ml-classification-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates .pkl files and test_data.csv)
python train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## 🌐 Live Streamlit App

🔗 **[ml-classification-app-akash.streamlit.app](https://ml-classification-app-akash.streamlit.app/)**

**App Features:**
- 📂 CSV upload for custom test data
- 🤖 Model selection dropdown (5 models)
- 📐 6 evaluation metrics display
- 🔲 Confusion matrix & ROC curve
- 📊 All-models comparison table & charts
- 🔍 Feature importance visualisation

---

## 📋 Requirements

```
streamlit==1.35.0
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
```

---

*Submitted for BITS Pilani NSP4 ML Assignment 2*
