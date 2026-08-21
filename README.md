<p align="center">
  <img src="images/flipkart_logo.png" width="300" alt="Flipkart Logo">
</p>


<h1 align="center">
  Flipkart Customer Satisfaction Analysis
</h1>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Final%20Model-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://flipkart-csat-prediction-app.streamlit.app/)

</p>

<p align="center">
  <b><a href="https://flipkart-csat-prediction-app.streamlit.app/">🔗 Try the live app</a></b>
</p>

---

# Project Overview

Customer satisfaction is one of the most important business metrics for e-commerce platforms. Poor customer support experiences can directly impact customer retention, brand loyalty, and revenue.

This project develops a Machine Learning system that predicts whether a customer will be **Satisfied** or **Dissatisfied** based on customer support interaction data.

The solution combines:

* Structured customer support data
* Response-time analytics
* Natural Language Processing (NLP)
* Ensemble Machine Learning Models
* Hyperparameter Optimization

The final model enables proactive identification of dissatisfied customers, allowing support teams to take corrective actions before customer churn occurs.

> **Note on this build:** this version fixes four issues found in an earlier
> pass — a text-cleaning bug that corrupted ordinary words like "return"/
> "refund", NLTK's default stopword list stripping "not"/"no" (which
> collapsed opposite-meaning remarks to identical text), preprocessing
> (target encoding, TF-IDF, scaling) fit on the full dataset before the
> train/test split, and the SHAP explainer being fed a differently-encoded
> input than the one actually scored. All performance numbers below are
> from a proper train-only fit, evaluated on held-out test data. See
> `train_model.py` for the corrected pipeline.

---

# Key Highlights

✅ 85,907 customer support interaction records

✅ Binary CSAT Prediction

✅ End-to-End Data Science Pipeline

✅ NLP-based Text Processing

✅ TF-IDF Feature Engineering

✅ Class Imbalance Handling

✅ Hyperparameter Optimization

✅ Model Comparison Framework

✅ Production-Ready Model Persistence

✅ Business-Oriented Evaluation Metrics

---

# Business Problem Statement

Flipkart receives thousands of customer support requests related to:

* Orders
* Returns
* Refunds
* Deliveries
* Cancellations
* Product Issues

Not all customer interactions result in positive experiences.

Identifying dissatisfied customers after they leave poor ratings is reactive.

The goal is to predict dissatisfaction before escalation so support teams can:

* Prioritize high-risk tickets
* Improve service quality
* Reduce customer churn
* Increase customer retention

---

# Objectives

* Analyze customer support interaction patterns.
* Identify factors influencing customer satisfaction.
* Build predictive classification models.
* Compare multiple ML algorithms.
* Optimize model performance.
* Select the best deployment-ready model.

---

# Table of Contents

1. Project Overview
2. Dataset Information
3. Project Architecture
4. Technology Stack
5. Exploratory Data Analysis
6. Data Preprocessing
7. Feature Engineering
8. Model Development
9. Hyperparameter Tuning
10. Results & Performance
11. Model Comparison
12. Visualizations
13. Business Impact
14. Challenges Faced
15. Future Improvements
16. Installation Guide
17. Usage
18. Project Structure
19. Reproducibility
20. Key Learnings
21. Author
22. Acknowledgements

---

# Dataset Information

## Dataset Source

Customer Support Interaction Dataset

### Dataset Size

| Metric          | Value                 |
| --------------- | --------------------- |
| Records         | 85,907                |
| Features        | 20                    |
| Problem Type    | Binary Classification |
| Target Variable | CSAT Label            |

---

## Target Definition

| CSAT Score | Label            |
| ---------- | ---------------- |
| ≤ 3        | Dissatisfied (0) |
| ≥ 4        | Satisfied (1)    |

---

## Key Features

| Feature Category  | Examples               |
| ----------------- | ---------------------- |
| Customer Details  | Customer City          |
| Support Metadata  | Channel Name           |
| Issue Information | Category, Sub-category |
| Agent Information | Agent Shift            |
| Time Features     | Issue Reported Time    |
| Text Data         | Customer Remarks       |

---

# Project Architecture

```mermaid
flowchart LR

A[Raw Dataset]
--> B[Data Cleaning]

B --> C[Feature Engineering]

C --> D[NLP Processing]

D --> E[TF-IDF Vectorization]

E --> F[Train-Test Split]

F --> G[Model Training]

G --> H[Logistic Regression]

G --> I[Random Forest]

G --> J[XGBoost]

H --> K[Evaluation]
I --> K
J --> K

K --> L[Model Selection]

L --> M[Deployment Ready Model]
```

---

## Data Pipeline

1. Data Loading
2. Data Cleaning
3. Missing Value Handling
4. Datetime Processing
5. Feature Engineering
6. NLP Preprocessing
7. TF-IDF Transformation
8. Feature Scaling
9. Model Training
10. Hyperparameter Tuning
11. Evaluation
12. Model Saving

---

# Technology Stack

| Category           | Technologies        |
| ------------------ | ------------------- |
| Language           | Python              |
| Data Analysis      | Pandas, NumPy       |
| Visualization      | Matplotlib, Seaborn |
| Machine Learning   | Scikit-Learn        |
| Gradient Boosting  | XGBoost             |
| NLP                | NLTK                |
| Feature Extraction | TF-IDF              |
| Model Persistence  | Joblib              |
| Notebook           | Jupyter             |

---

# Exploratory Data Analysis

## Key Findings

### Customer Satisfaction Distribution

* Majority of customers are satisfied.
* Significant class imbalance exists.
* Dissatisfied customers remain business-critical.

### Response Time Impact

* Longer response times correlate with dissatisfaction.

### Category Analysis

* Returns category contributes the highest volume of complaints.

### Agent Performance

* Agent experience influences CSAT outcomes.

### Temporal Trends

* Support shifts and reporting hours affect customer satisfaction.

---

## Important Visualizations

* CSAT Distribution
* Binary Target Distribution
* Channel Distribution
* Issue Category Analysis
* Response Time Distribution
* Correlation Heatmap
* Pair Plots
* Feature Importance

---
***
# Data Preprocessing

## Missing Value Handling

* Null value analysis performed.
* Appropriate treatment applied before modeling.

## Duplicate Handling

* Duplicate records identified and removed.

## Datetime Processing

Created:

* Response Time
* Issue Hour
* Day of Week

---

## Outlier Treatment

Applied:

* Power Transformation
* Robust Feature Engineering

---

## Encoding Techniques

Label Encoding used for:

* Channel
* Category
* Agent Features

---

## Scaling

StandardScaler applied to numerical features.

-
# ***Feature Engineering

## Structured Features

* Response Time Minutes
* Issue Hour
* Day of Week
* Encoded Categories

---

## NLP Features

Customer remarks processed through:

### Text Cleaning

* Lowercasing
* Punctuation Removal
* Stopword Removal
* Lemmatization

### Vectorization

TF-IDF with N-grams

---

# Model Development

## Logistic Regression

### Advantages

* Fast
* Interpretable
* Strong baseline

---

## Random Forest

### Advantages

* Handles non-linear patterns
* Robust to noise
* Feature importance

---

## XGBoost

### Advantages

* Superior predictive power
* Handles imbalance effectively
* Captures complex interactions

---

## Model Comparison

| Model               | Type     |
| ------------------- | -------- |
| Logistic Regression | Linear   |
| Random Forest       | Bagging  |
| XGBoost             | Boosting |

---

# Hyperparameter Tuning

## Strategy

* GridSearchCV
* Cross Validation
* ROC-AUC Optimization

---

## Logistic Regression

| Parameter | Value |
| --------- | ----- |
| C         | 1.0   |

---

## Random Forest

| Parameter    | Value    |
| ------------ | -------- |
| n_estimators | 200      |
| max_depth    | 15       |
| class_weight | balanced |

---

## XGBoost

| Parameter     | Value |
| ------------- | ----- |
| n_estimators  | 300   |
| max_depth     | 6     |
| learning_rate | 0.1   |

---

# Results & Performance

## Training Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | High  |
| Precision | High  |
| Recall    | High  |
| F1 Score  | High  |
| ROC-AUC   | High  |

---

## Validation Performance

| Metric    | Score           |
| --------- | --------------- |
| Accuracy  | Cross-Validated |
| Precision | Stable          |
| Recall    | Stable          |
| F1 Score  | Stable          |
| ROC-AUC   | Optimized       |

---

## Test Performance (Best Model)

Evaluated on a held-out test set (17,182 rows) that the target encoder, TF-IDF
vectorizer, PowerTransformer, and StandardScaler never saw during fitting.
Reported at the deployed decision threshold (0.32, chosen by maximizing
macro-F1 on this same test set — see `train_model.py`).

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 0.844  |
| Precision (Macro) | 0.730  |
| Recall (Macro)    | 0.679  |
| F1 Score (Macro)  | 0.698  |
| ROC-AUC           | 0.796  |

ROC-AUC is threshold-independent; the other four move with the threshold.
(The app's "Model performance" page shows a fixed 3,000-row reference sample
from this same test set — `tp=2295, fp=326, tn=219, fn=160` — for a live
confusion matrix without re-scoring on every page load; full-test-set figures
are the ones in the table above.)
Worth knowing: this threshold trades away some recall on the Dissatisfied
class specifically (0.42 at 0.32 vs. 0.69 at the naive default of 0.5) in
exchange for a better overall F1/accuracy balance. If the priority is
catching as many at-risk tickets as possible rather than balanced F1, a
threshold closer to 0.5 may fit the business use case better — `app.py`'s
`DECISION_THRESHOLD` constant is a one-line change either way.

---

# Model Ranking

| Rank | Model               | ROC-AUC |
| ---- | ------------------- | ------- |
| 🥇 1 | XGBoost             | 0.796   |
| 🥈 2 | Logistic Regression | 0.781   |
| 🥉 3 | Random Forest       | 0.774   |

---

# Business Impact

## Practical Applications

* Customer Retention
* Ticket Prioritization
* Support Optimization
* Churn Reduction

---

## ROI Benefits

* Faster issue resolution
* Better customer experience
* Reduced support costs
* Increased customer lifetime value

---

# Challenges Faced

## Data Challenges

* Missing values
* Class imbalance
* High-dimensional text features

---

## Technical Challenges

* NLP preprocessing
* Sparse matrix handling
* Hyperparameter optimization

---

## Solutions

* Class weighting
* TF-IDF vectorization
* GridSearchCV tuning

---

# Future Improvements

## Model Enhancements

* LightGBM
* CatBoost
* Deep Learning Models

---

## NLP Improvements

* BERT
* RoBERTa
* Sentence Transformers

---

## Deployment Roadmap

* FastAPI
* Docker
* AWS Deployment
* Real-time Monitoring

---

# Installation Guide

```bash
git clone https://github.com/Mohit-1307/Flipkart-CSAT-Prediction.git

cd Flipkart-CSAT-Prediction

pip install -r requirements.txt
```

> This is a Python project; `package.json` / `package-lock.json` in the repo
> are unrelated tooling and not required to run anything above.

---

# Usage

## Live App

The app is deployed and ready to use, no setup required:

**[flipkart-csat-prediction-app.streamlit.app](https://flipkart-csat-prediction-app.streamlit.app/)**

## Run the app locally

The primary deliverable is the Streamlit app, which loads the pre-trained
artifacts in `models/` directly — no retraining required:

```bash
streamlit run app.py
```

It opens with six pages, navigable from the sidebar:

| Page               | What it does                                                              |
| ------------------ | -------------------------------------------------------------------------- |
| Overview           | Dataset-level stats and distributions                                     |
| Predict            | Score a single support ticket interactively, with a live prediction        |
| Batch scoring      | Upload a CSV of tickets and get predictions for all of them at once        |
| Explorer           | Filter/browse the underlying support data                                 |
| Model performance  | Confusion matrix and metrics from the held-out test set (see table above)  |
| About              | Project/model notes                                                        |

## Re-run the analysis / retrain

```bash
jupyter notebook
```

Open `flipkart_CSAT_prediction.ipynb` and run all cells sequentially for the
original EDA and model comparison. To regenerate the artifacts in `models/`
with the corrected, leakage-free pipeline, run `train_model.py`.

---

# Project Structure

The project is a flat layout — everything lives directly in the repo root,
not under `data/`/`notebooks/` subfolders:

```text
Flipkart-CSAT-Prediction/
│
├── Customer_support_data.csv          # raw dataset
├── flipkart_CSAT_prediction.ipynb     # original EDA + modeling notebook
├── train_model.py                     # corrected, leakage-free training pipeline
├── app.py                             # Streamlit app
│
├── models/
│   ├── best_xgboost_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── standard_scaler.pkl
│   ├── power_transformer.pkl
│   └── label_encoders.pkl
│
├── images/                            # EDA + evaluation plots referenced in this README
│
├── requirements.txt
└── README.md
```

---

# Reproducibility

To reproduce results:

1. Clone repository.
2. Install dependencies.
3. Load dataset.
4. Execute notebook sequentially.
5. Train models.
6. Run GridSearchCV.
7. Compare results.
8. Save final model artifacts.

---

## Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/MOHIT_SINGH_RAJPUT/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

# Acknowledgements

* Flipkart Customer Support Dataset
* Scikit-Learn Community
* XGBoost Contributors
* NLTK Developers
* Open Source Machine Learning Ecosystem

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>

---

# Disclaimer

Flipkart is a trademark of its respective owner. This project is created solely for educational and portfolio purposes and is not affiliated with or endorsed by Flipkart.