# 💳 Credit Card Fraud Detection — ML Engineering Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🎯 Problem Statement

Every second, thousands of credit card transactions happen worldwide. A tiny fraction are fraudulent — but that fraction costs billions annually.

This project builds and compares multiple ML models to detect fraudulent transactions from 284,807 real European credit card transactions — where only 0.17% are fraud.

The core challenge: **severe class imbalance.**
A model that predicts "legitimate" for every transaction scores 99.83% accuracy — while catching zero fraudsters. This project addresses that challenge head-on.

---

## 📊 Dataset

- **Source:** ULB Machine Learning Group  
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- **Size:** 284,807 transactions | 31 features  
- **Fraud cases:** 492 (0.17%)  
- **Features:** V1–V28 (PCA anonymised), Amount, Time, Class  

---

## 🔬 What This Project Covers

### 1. Baseline Models
Three algorithms trained with no imbalance handling:
- Decision Tree
- Random Forest
- XGBoost

**Finding:** All three gave identical Recall (78.57%) — proving the problem was the data, not the algorithm.

### 2. Imbalance Handling
Two techniques applied and compared:

**Class Weights** (`class_weight='balanced'`)
- Tells the model fraud cases matter more during training
- One line of code
- Recall jumped from 78.57% → 84.69%

**SMOTE** (Synthetic Minority Oversampling Technique)
- Generates synthetic fraud cases from real patterns
- Balanced training data: 227,451 fraud vs 227,451 legitimate
- Recall jumped to 88.78% (RF) and 89.80% (XGBoost)
- Applied on training data only — never on test data

### 3. Business Decision Framework
Built a scenario-based recommendation system that:
- Takes business priority as input
- Returns the right model with full business impact
- Calculates money saved, money lost, and net benefit

---

## 📈 Results

| Model | Recall | Precision | F1 | Caught | Missed | False Alarms |
|---|---|---|---|---|---|---|
| Decision Tree | 78.57% | 84.62% | 81.48% | 77/98 | 21 | 14 |
| Random Forest | 78.57% | 95.06% | 86.03% | 77/98 | 21 | 4 |
| XGBoost | 78.57% | 91.67% | 84.62% | 77/98 | 21 | 7 |
| RF + Class Weights | 84.69% | 64.84% | 73.45% | 83/98 | 15 | 45 |
| RF + SMOTE | 88.78% | 35.95% | 51.18% | 87/98 | 11 | 155 |
| **XGBoost + SMOTE** | **89.80%** | **27.41%** | **42.00%** | **88/98** | **10** | **233** |

---

## 💰 Business Impact Analysis

### Scenario 1 — Small Bank / Fintech Startup
```
Priority:           Customer experience
Recommended model:  Random Forest (baseline)
Fraudsters caught:  77 out of 98
Money saved:        €38,500
False alarm costs:  €20
Net benefit:        €27,980
```

### Scenario 2 — Large Bank / Payment Gateway
```
Priority:           Maximum fraud prevention
Recommended model:  XGBoost + SMOTE
Fraudsters caught:  88 out of 98
Money saved:        €44,000
False alarm costs:  €1,165
Net benefit:        €37,835
```

### Scenario 3 — Mid-size Bank / Insurance Company
```
Priority:           Balanced approach
Recommended model:  Random Forest + Class Weights
Fraudsters caught:  83 out of 98
Money saved:        €41,500
False alarm costs:  €225
Net benefit:        €33,775
```

---

## 🧠 Key Learnings

**1. Accuracy is not enough on imbalanced data**  
99.83% accuracy means nothing when 0 fraudsters are caught. Always use Precision, Recall, and F1 on imbalanced datasets.

**2. The problem was data, not algorithms**  
All three baseline models gave identical Recall. Changing the algorithm never fixes a data problem.

**3. The Precision-Recall tradeoff is real**  
Every improvement in Recall came at the cost of Precision. There is no free lunch — only business tradeoffs.

**4. There is no single best model**  
Model selection depends entirely on business priority. The right model for a startup is wrong for a large bank.

**5. SMOTE on training data only — always**  
Test data must represent the real world. Synthetic data in the test set produces fake results and dangerous decisions.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas | Data manipulation |
| Scikit-learn | ML models, metrics, train/test split |
| XGBoost | Gradient boosting |
| Imbalanced-learn | SMOTE oversampling |
| Matplotlib + Seaborn | Visualisation |

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection-ML/
│
├── notebooks/
│   ├── Day1_Baseline_Models.ipynb
│   └── Day2_SMOTE_ClassWeights.ipynb
│
├── data/
│   └── README.md        ← download link (file too large for GitHub)
│
└── README.md
```

---

## 👩‍💻 Author

**Manpreet Kaur**  
MS Data Science | AI/ML Engineer  


---

*Real dataset. Real business decisions. Real engineering.*
