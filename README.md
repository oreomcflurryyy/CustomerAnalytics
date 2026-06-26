# 📊 Customer Churn Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts customer churn using the IBM Telco Customer Churn dataset. The project demonstrates the complete data science workflow—from data cleaning and exploratory data analysis to feature engineering, model training, evaluation, and model persistence.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Automated ML Pipeline
- Logistic Regression
- Random Forest
- Model Comparison
- Saved Trained Models
- Automatic Visualization Generation
- Modular Project Structure

---

## 📂 Project Structure

```text
CustomerAnalytics/

├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── outputs/
│   ├── churn_distribution.png
│   ├── contract_vs_churn.png
│   ├── paymentmethod_vs_churn.png
│   ├── ...
│   └── model_comparison.csv
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📈 Workflow

```text
              Raw Dataset
                   │
                   ▼
          Data Inspection
                   │
                   ▼
           Data Cleaning
                   │
                   ▼
     Exploratory Data Analysis
                   │
                   ▼
        Feature Engineering
                   │
                   ▼
         Train/Test Split
                   │
                   ▼
     Preprocessing Pipeline
(StandardScaler + OneHotEncoder)
                   │
                   ▼
        Machine Learning
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
 Logistic Regression   Random Forest
      │                     │
      └──────────┬──────────┘
                 ▼
        Model Evaluation
                 │
                 ▼
      Save Models & Reports
```

---

# 📊 Dataset

**Dataset:** IBM Telco Customer Churn

| Attribute | Value |
|----------|------:|
| Samples | 7043 |
| Features | 21 |
| Target | Churn |
| Missing Rows Removed | 11 |

Target Variable:

- Yes → Customer Churned
- No → Customer Stayed

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs:

- Missing value handling
- Duplicate checking
- Data type correction
- Feature engineering
- One-Hot Encoding
- Feature Scaling
- Train/Test Split

---

# 📊 Exploratory Data Analysis

## Customer Churn Distribution

![](outputs/churn_distribution.png)

---

## Contract vs Churn

![](outputs/contract_vs_churn.png)

---

## Internet Service vs Churn

![](outputs/internetservice_vs_churn.png)

---

## Payment Method vs Churn

![](outputs/paymentmethod_vs_churn.png)

---

# 🤖 Machine Learning Models

Implemented models:

- Logistic Regression
- Random Forest

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|-------:|---------:|---------:|
| Logistic Regression | **80.38%** | **64.85%** | **57.22%** | **60.80%** | **0.836** |
| Random Forest | 79.18% | 63.11% | 52.14% | 57.10% | 0.830 |

### Best Performing Model

**Logistic Regression**

Accuracy: **80.38%**

ROC-AUC: **0.836**

---

# 💡 Key Insights

Exploratory Data Analysis revealed:

- Month-to-month contracts have the highest churn rate.
- Electronic Check customers churn significantly more.
- Fiber Optic customers churn more than DSL customers.
- Customers without Online Security are much more likely to churn.
- Customers without Tech Support have higher churn.
- Gender has minimal impact on churn.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/oreomcflurryyy/CustomerAnalytics.git
```

Move into the project:

```bash
cd CustomerAnalytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

# 📁 Generated Outputs

Running the pipeline automatically creates:

```
models/
    logistic_regression.pkl
    random_forest.pkl

outputs/
    *.png
    model_comparison.csv
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Git
- GitHub

---

# 🚀 Future Improvements

- XGBoost
- LightGBM
- Hyperparameter Optimization
- SHAP Explainability
- Streamlit Dashboard
- Docker Support
- CI/CD Pipeline

---

# 👩‍💻 Author

**Eshanika Dey**

B.Tech, Bioscience & Biotechnology

Indian Institute of Technology Kharagpur

Interested in:
- Machine Learning
- Data Science
- Bioinformatics
- Computational Biology

---

## ⭐ If you found this project useful, consider starring the repository!
