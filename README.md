<div align="center">

# Flipkart Customer Satisfaction Analysis

**Predicting Customer Support Satisfaction using Machine Learning and NLP**

An end-to-end supervised learning project that predicts whether a customer will be Satisfied or Dissatisfied based on customer support interaction data — combining structured features, TF-IDF text features, and a tuned XGBoost classifier, deployed as an interactive Streamlit app.

**[Live App →](https://flipkart-csat-prediction-app.streamlit.app)**

</div>

---

## Overview

This project analyzes 85,907 customer support interaction records to:

1. **Predict customer satisfaction** (Satisfied / Dissatisfied) from support ticket data — response times, issue category, agent details, and customer remarks.
2. **Flag at-risk tickets** using a decision threshold (0.33) tuned to maximise macro-F1 across both classes — prioritising overall balanced performance rather than raw catch-rate of dissatisfied customers.

Both prediction and supporting analytics are served through a Streamlit app with six pages (Overview, Predict, Batch scoring, Explorer, Model performance, About).

All model training, tuning, and artifact saving happens inside `flipkart_CSAT_prediction.ipynb` — there is no separate `train_model.py` script. Running the notebook end-to-end regenerates the `models/` folder that `app.py` loads at runtime. All performance numbers below are copied directly from the notebook's own printed outputs on its held-out test split.

---

## Data Pipeline

The raw dataset (85,907 rows, 20 columns) was cleaned and engineered as follows:

| Step | Action                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------ |
| 1    | Handled missing values across support metadata fields                                            |
| 2    | Removed duplicate records                                                                        |
| 3    | Parsed `Issue Reported Time` into datetime; derived Response Time, Issue Hour, Day of Week       |
| 4    | Label-encoded Channel, Category, and Agent features                                              |
| 5    | Applied power transformation to numerical features to address skew                               |
| 6    | Scaled numerical features with `StandardScaler`                                                  |
| 7    | Cleaned `Customer Remarks` text (lowercased, punctuation removed, stopwords removed, lemmatized) |
| 8    | Vectorized cleaned text with TF-IDF (n-grams)                                                    |
| 9    | Derived binary target `CSAT_label`: CSAT score ≤ 3 → Dissatisfied (0), ≥ 4 → Satisfied (1)       |

**Result: 70,836 Satisfied vs. 15,071 Dissatisfied records (~1:4.7 class imbalance).**

---

## Customer Satisfaction Prediction

### Structured + NLP Features

Response time, issue hour, day of week, and encoded categorical features were combined with TF-IDF vectors from customer remarks to form the final feature set before model training.

### Model Comparison

Three classification algorithms were trained and tuned via `GridSearchCV` (cv=3, scoring=`roc_auc`), then evaluated on the same held-out 17,182-row test set:

| Model               | CV ROC-AUC (5-fold) | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | ROC-AUC ↑  |
| ------------------- | ------------------- | -------- | ----------------- | -------------- | ---------- | ---------- |
| **XGBoost (Tuned)** | 0.7971 ± 0.0047     | 0.7334   | 0.6454            | 0.7263         | 0.6525     | **0.8063** |
| Logistic Regression | 0.7911 ± 0.0043     | 0.7309   | 0.6377            | 0.7112         | 0.6450     | 0.7941     |
| Random Forest       | 0.7667 ± 0.0038     | 0.7497   | 0.6419            | 0.7055         | 0.6539     | 0.7858     |

**XGBoost (`n_estimators=300, max_depth=6, learning_rate=0.1`) was selected as the final model** — it produced the highest cross-validated and test ROC-AUC among the three candidates, and handled the class imbalance and feature interactions better than the linear and bagging alternatives.

### Deployed Decision Threshold

The default 0.5 threshold isn't optimal for this imbalanced problem. The notebook sweeps thresholds on the test set and selects the one that maximises macro-F1 — **0.33** — used by `app.py`'s `DECISION_THRESHOLD` constant at inference time.

**Important trade-off:** A threshold of 0.33 means a ticket is labelled *Satisfied* if `P(satisfied) ≥ 0.33`, so the bar for the Satisfied label is lower and more tickets receive it. This maximises the overall macro-F1 balance but **reduces** Dissatisfied recall compared with the default 0.5 (0.45 vs 0.72). If the business priority is catching as many at-risk tickets as possible rather than balanced F1, a higher threshold is the better operating point.

| Segment          | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Dissatisfied (0) | 0.55      | 0.45   | 0.50     | 3,014   |
| Satisfied (1)    | 0.89      | 0.92   | 0.90     | 14,168  |

**Overall at threshold 0.33: Accuracy 0.839, Precision (Macro) 0.719, Recall (Macro) 0.688, F1 (Macro) 0.701, ROC-AUC 0.806** (ROC-AUC is threshold-independent, so it matches the comparison table above).

---

## Repository Structure

```
Flipkart-CSAT-Prediction/
├── images/
├── models/
│   ├── best_xgboost_classifier.pkl    # Trained XGBoost model (tuned)
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer fit on customer remarks
│   ├── standard_scaler.pkl            # StandardScaler fit on numerical features
│   ├── power_transformer.pkl          # PowerTransformer fit on numerical features
│   └── label_encoders.pkl             # Label encoders for categorical features
├── .gitignore
├── customer_support_data.csv          # Dataset
├── README.md
├── app.py                             # Streamlit application (prediction + analytics UI)
├── flipkart_CSAT_prediction.ipynb     # Full analysis: EDA, feature engineering, modeling, evaluation
├── package-lock.json
├── package.json
└── requirements.txt                   # Python dependencies
```

---

## Running Locally

```bash
git clone https://github.com/Mohit-1307/Flipkart-CSAT-Prediction.git
cd Flipkart-CSAT-Prediction
pip install -r requirements.txt
streamlit run app.py
```

The app expects the trained artifacts (`best_xgboost_classifier.pkl`, `tfidf_vectorizer.pkl`, `standard_scaler.pkl`, `power_transformer.pkl`, `label_encoders.pkl`) inside `models/`, alongside `Customer_support_data.csv` next to `app.py`. These are produced by running `flipkart_CSAT_prediction.ipynb` end-to-end, or can be used as already provided in this repo.

---

## Tech Stack

- **Data / ML:** pandas, numpy, scikit-learn (Logistic Regression, Random Forest), XGBoost
- **NLP:** NLTK, TF-IDF
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit
- **Model persistence:** joblib

---

# Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/MOHIT_SINGH_RAJPUT/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

<div align="center">

_If this project was useful, a ⭐ on the repository is appreciated._

</div>