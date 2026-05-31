# 💳 Credit Risk Analysis & Prediction

A data analysis and machine learning project that explores the factors driving loan default risk, combining rigorous exploratory data analysis with interpretable ML models and a live Streamlit prediction dashboard.

> Built as a capstone project | Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit

---

## 📌 Project Overview

Financial institutions rely on credit risk models to decide whether to approve loans and at what interest rate. This project works through the full analytical pipeline — from raw data to an explainable, deployed model — on a real-world credit risk dataset of over 32,000 loan applicants.

The core question driving the analysis: **which applicant characteristics most strongly predict loan default, and how confidently can a model flag high-risk applicants before a lender commits?**

---

## 🔍 Key Findings from EDA

- **Loan grade is the single strongest categorical predictor of default** — Grade A applicants default at a fraction of the rate of Grade E–G applicants, suggesting lenders' own grading already encodes significant risk signal.
- **High loan-to-income ratios cluster heavily in defaults** — applicants allocating more than 30% of income to loan repayment show disproportionately high default rates, visible clearly in the `loan_percent_income` distribution.
- **Interest rate and loan grade are highly correlated** (~0.94 Pearson correlation), indicating multicollinearity that tree-based models handle better than linear ones.
- **Class imbalance is present** (~78% non-default vs ~22% default), making precision and recall more informative evaluation metrics than raw accuracy.
- **Renters default more frequently than mortgage holders**, likely reflecting underlying income and asset stability differences rather than home ownership per se.

---

## 🧱 Project Structure

```
Sikho/
├── Code/
│   ├── Credit Risk.ipynb          # Full analysis notebook
│   ├── app.py                     # Streamlit prediction dashboard
│   ├── credit_risk_dataset.csv    # Dataset (32,000+ records)
│   └── xgboost_model.pkl          # Serialised best model
└── README.md
```

---

## 🔬 Methodology

### 1. Data Cleaning
- Identified missing values in `person_emp_length` and `loan_int_rate`; imputed using column medians to preserve dataset size without introducing bias from row deletion
- Removed duplicate records after deduplication check
- Applied `LabelEncoder` to four categorical features: `person_home_ownership`, `loan_intent`, `loan_grade`, and `cb_person_default_on_file`

### 2. Exploratory Data Analysis
- **Distribution plots** for all categorical features to understand class balance across ownership types, loan intents, and grades
- **Countplots by loan status** to visually compare default rates across each category level
- **Boxplots** of all numerical features (age, income, employment length, loan amount, interest rate, loan-to-income ratio, credit history length) split by loan status — revealing that defaulters tend to have higher interest rates and lower incomes
- **Correlation heatmap** identifying feature relationships, notably the strong collinearity between interest rate and loan grade
- **Loan-to-income distribution** showing a right-skewed distribution with a heavy tail among defaulters

### 3. Modelling
Four classifiers trained on a standardised 80/20 train-test split:

| Model | Key Strength |
|---|---|
| Decision Tree | Interpretable baseline |
| Random Forest | Reduces overfitting via bagging |
| XGBoost | Best overall performance |
| MLP (Neural Network) | Captures non-linear feature interactions |

All models evaluated on **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** — with XGBoost achieving the strongest balance across metrics.

### 4. Explainability with SHAP
- SHAP (SHapley Additive exPlanations) applied to both the XGBoost and MLP models
- **Summary plots** rank features by global importance: `person_income`, `loan_grade`, and `loan_percent_income` dominate
- **Waterfall plots** explain individual predictions — showing exactly which features pushed a specific applicant's risk score up or down
- This explainability layer is critical in a lending context where regulators require justifiable decisions

---

## 🚀 Live Dashboard

The project includes a Streamlit app (`app.py`) that:
- Accepts loan applicant details via an interactive sidebar
- Returns a **real-time default risk prediction** with probability score
- Displays a **SHAP waterfall chart** explaining the specific prediction
- Shows **global feature importance** across all applicants

**To run locally:**
```bash
pip install streamlit xgboost shap scikit-learn pandas numpy matplotlib
streamlit run Code/app.py
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Data Manipulation | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn (Decision Tree, Random Forest, MLP), XGBoost |
| Explainability | SHAP |
| Deployment | Streamlit |
| Environment | Jupyter Notebook, Python 3.x |

---

## 📂 Dataset

The dataset contains **32,000+ loan records** with the following features:

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income (ZAR / USD) |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of loan (Personal, Education, Medical, etc.) |
| `loan_grade` | Lender-assigned credit grade (A–G) |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate assigned |
| `loan_percent_income` | Loan amount as % of income |
| `cb_person_default_on_file` | Prior default on credit bureau record (Y/N) |
| `cb_person_cred_hist_length` | Length of credit history in years |
| `loan_status` | **Target** — 0: No Default, 1: Default |

---

## 📈 Results Summary

XGBoost outperformed the other classifiers, achieving strong recall on the minority (default) class — the most consequential metric for a lender trying to avoid approving high-risk applicants. SHAP analysis confirmed that the model's top three drivers (`person_income`, `loan_grade`, `loan_percent_income`) align with established credit risk theory, which increases confidence in the model's real-world reliability.

---

## 🔧 Potential Improvements

- Hyperparameter tuning with `GridSearchCV` or `Optuna` for XGBoost
- SMOTE oversampling to address class imbalance
- Cross-validation to improve generalisation estimates
- Feature engineering: debt-to-income bins, age-employment interaction term
- Deploy Streamlit app to Streamlit Cloud for public access

---

## 👤 Author

**Sikelelä Mda**  
[github.com/SikelelaSomp/Sikho](https://github.com/SikelelaSomp/Sikho)

---

*This project was developed as an academic capstone and is intended for educational and portfolio purposes.*
