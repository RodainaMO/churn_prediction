# 📊 Customer Churn Prediction

> End-to-end machine learning project with an interactive Streamlit dashboard  
> **Stack:** Python · Scikit-learn · XGBoost · SMOTE · Streamlit · Plotly

---

## 🎯 Problem Statement

Customer churn is one of the most costly challenges for subscription-based businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a production-grade ML pipeline to identify at-risk customers **before they leave**, enabling targeted retention campaigns.

---

## 🏗️ Project Structure

```
churn-prediction/
├── Churn_Prediction_Notebook.ipynb   ← Full ML pipeline notebook
├── app.py                            ← Streamlit web application
├── churn_model.pkl                   ← Trained model (auto-generated)
├── requirements.txt                  ← Python dependencies
└── README.md
```

---

## 📋 Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features  
- Demographics (gender, senior citizen, dependents)
- Services (phone, internet, streaming, tech support)
- Billing (contract type, payment method, monthly charges)
- **Target:** Churn (Yes/No)

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Target imbalance analysis (26% churn rate)
- Feature distributions and churn rates by segment
- Correlation analysis

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `AvgMonthlySpend` | TotalCharges / (tenure + 1) |
| `HasMultiServices` | Count of active services |
| `IsSeniorLongTenure` | Senior citizen with tenure > 24 months |
| `TenureBucket` | Binned tenure (0-1yr, 1-2yr, 2-4yr, 4+yr) |

### 3. Model Comparison (5-Fold CV)
| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.83 |
| Random Forest | ~0.84 |
| Gradient Boosting | ~0.85 |
| **XGBoost** ✅ | **~0.86** |

### 4. Class Imbalance → SMOTE
Applied Synthetic Minority Over-sampling Technique (SMOTE) in the pipeline to handle the 3:1 class imbalance.

### 5. Hyperparameter Tuning
GridSearchCV over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`.

---

## 📈 Results

| Metric | Score |
|---|---|
| **ROC-AUC** | **~0.86** |
| Average Precision | ~0.65 |
| Recall (Churn class) | ~0.80 |
| Algorithm | XGBoost + SMOTE |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will auto-download the dataset and train the model on first run (~30 seconds).

---

## 🖥️ Streamlit App Features

| Tab | Features |
|---|---|
| **Single Prediction** | Enter one customer profile → get churn probability + gauge chart + retention recommendations |
| **Batch Prediction** | Upload a CSV → get predictions for all rows with risk segmentation + downloadable results |
| **Model Insights** | Feature importances, churn rate by segment, tenure distribution, business insights |

---

## 💡 Key Business Insights

1. **Contract type** is the strongest predictor — month-to-month customers churn at 43% vs <5% for 2-year contracts
2. **First 12 months** are the highest-risk period for churn
3. **Fiber optic** users churn significantly more than DSL users
4. **No tech support** is a strong churn indicator
5. High **monthly charges** correlate with increased churn risk

---

## 📦 Requirements

```
streamlit>=1.32
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
plotly>=5.18
matplotlib>=3.7
seaborn>=0.12
```

---


