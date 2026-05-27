# 🤖 Machine Learning Projects Portfolio

> 4 end-to-end ML projects built from scratch using Python & scikit-learn.
> No AutoML libraries used — everything implemented manually.

---

## 📁 Projects Overview

| # | Project | Type | Algorithm | Language |
|---|---------|------|-----------|----------|
| 1 | [Sales Forecasting](#1-sales-forecasting-using-ml) | Regression | Linear Regression, Random Forest, Gradient Boosting | Python |
| 2 | [Genetic Algorithms](#2-genetic-algorithms-for-optimization) | Optimization | GA from scratch | Python |
| 3 | [Multi-Label Text Categorization](#3-multi-label-text-categorization) | NLP / Classification | TF-IDF + OneVsRest | Python |
| 4 | [AutoML Framework](#4-automl-framework) | Framework | 9–10 models + GridSearchCV | Python |

---

## 1. Sales Forecasting using ML

📂 Folder: `sales_forecasting/`

### What it does
Predicts future monthly sales using historical data.
Trained on **43 months** of sales data and forecasts up to **2026**.

### ML Pipeline
```
Raw Sales Data → Feature Engineering → Train/Test Split → Model Training → Forecast
```

### Models Used
| Model | R² Score |
|-------|----------|
| Linear Regression | 1.00 |
| Random Forest | -4.34 |
| Gradient Boosting | -5.29 |

### Features Engineered
- Lag features (1, 2, 3 months)
- Rolling averages (3-month, 6-month)
- Festival/season flags (Diwali, year-end)
- Month, Quarter, Year

### Key Results
- 📈 Best Model: **Linear Regression** (R² = 1.0)
- 🔮 2025 Avg Forecast: **₹1.44L/month**
- 🔮 2026 Avg Forecast: **₹1.62L/month**
- 🏆 Most Important Feature: Rolling 3-month avg (63.4%)

### How to Run
```bash
cd sales_forecasting
pip install -r requirements.txt
python sales_forecast.py
```

### Output
```
✅ Best Model: Linear Regression
📁 results.json saved
```

---

## 2. Genetic Algorithms for Optimization

📂 Folder: `genetic_algorithms/`

### What it does
Solves 3 classic optimization problems using Genetic Algorithms built from scratch — no ML library needed for the GA logic.

### Problems Solved

| Problem | Goal | Result |
|---------|------|--------|
| Function Optimization | Maximize f(x) = sin(x)·cos(x/2) + x/10 | x = 17.69, f(x) = 2.54 |
| Knapsack Problem | Best items within 10 kg limit | ₹550 value, 10/10 kg used |
| Travelling Salesman (TSP) | Shortest route across 10 Indian cities | 5693 km |

### GA Concepts Implemented
- **Population** — random initial solutions
- **Fitness Function** — how good is each solution
- **Tournament Selection** — best of 3 random individuals
- **Crossover** — single-point (Function/Knapsack), Order Crossover OX (TSP)
- **Mutation** — bit flip (binary), swap mutation (TSP)
- **Generations** — evolves over 150–500 generations

### TSP Route Found
```
Ahmedabad → Jaipur → Delhi → Kolkata → Vijayawada →
Hyderabad → Chennai → Bangalore → Pune → Mumbai → Ahmedabad
```

### How to Run
```bash
cd genetic_algorithms
pip install -r requirements.txt
python genetic_algorithm.py
```

### Output
```
✅ [1/3] Function Optimization — Best f(x) = 2.535906
✅ [2/3] Knapsack — Total Value = ₹550
✅ [3/3] TSP — Distance = 5693.46 km
📁 results.json saved
```

---

## 3. Multi-Label Text Categorization

📂 Folder: `multilabel_text/`

### What it does
Classifies news headlines into **multiple categories simultaneously**.
One headline can belong to more than one category at the same time.

**Example:**
```
"India wins cricket World Cup final"
→ Labels: [Sports ✓, India ✓]

"ISRO launches satellite for climate study"
→ Labels: [Science ✓, India ✓, Technology ✓]
```

### Categories (8 total)
`Politics` · `Sports` · `Technology` · `Business` · `Health` · `Entertainment` · `Science` · `India`

### ML Pipeline
```
200 Headlines → Text Cleaning → TF-IDF Vectorization (bigrams)
→ MultiLabelBinarizer → OneVsRest Classifier → Multi Labels
```

### Models Compared
| Model | Hamming Loss ↓ | F1-Micro |
|-------|---------------|---------|
| Logistic Regression | 0.2344 | 0.4361 |
| Linear SVM | 0.2344 | 0.4361 |
| Random Forest | 0.2344 | 0.4361 |

### Evaluation Metrics Explained
| Metric | Meaning |
|--------|---------|
| **Hamming Loss** | Lower = better. % of wrong labels per sample |
| **F1-Micro** | Overall F1 treating each label prediction equally |
| **F1-Macro** | Average F1 per category (handles imbalance better) |
| **Subset Accuracy** | Exact match — every label must be correct |

### How to Run
```bash
cd multilabel_text
pip install -r requirements.txt
python multilabel_categorization.py
```

### Output
```
✅ Model comparison table printed
✅ Predictions on 7 new headlines
📁 results.json saved
```

---

## 4. AutoML Framework

📂 Folder: `automl_framework/`

### What it does
A fully automated ML pipeline that:
- Profiles your data automatically
- Preprocesses (impute, scale, encode) automatically
- Selects top features automatically
- Trains 9–10 models and compares them
- Tunes hyperparameters with GridSearchCV
- Picks the best model and saves it as `.pkl`

### AutoML Pipeline
```
Your Data
   ↓
📊 Data Profiling      (shape, missing %, class balance)
   ↓
⚙️  Auto Preprocessing  (median impute → StandardScaler → OneHotEncode)
   ↓
🔍 Feature Selection   (SelectKBest → top 80% features)
   ↓
🤖 Model Selection     (trains all models, 5-fold CV)
   ↓
🎯 Hyperparameter Tune (GridSearchCV — best params)
   ↓
📈 Evaluation          (accuracy / R², F1, RMSE, ROC-AUC)
   ↓
💾 Save Best Model     (.pkl file)
```

### Models Trained Automatically

**Classification (9 models)**
Logistic Regression · Decision Tree · Random Forest · Gradient Boosting ·
Extra Trees · SVM · KNN · Naive Bayes · Ridge Classifier

**Regression (10 models)**
Linear Regression · Ridge · Lasso · ElasticNet · Decision Tree ·
Random Forest · Gradient Boosting · Extra Trees · SVR · KNN

### Results on 3 Demo Datasets

| Dataset | Task | Best Model | Score |
|---------|------|------------|-------|
| Iris Flowers (150 samples) | Classification | Random Forest | 96.67% CV Accuracy |
| Breast Cancer (569 samples) | Classification | Logistic Regression | 98.02% CV Accuracy |
| Diabetes (442 samples) | Regression | Lasso | R² = 0.47 |

### How to Run
```bash
cd automl_framework
pip install -r requirements.txt

# Run on 3 demo datasets
python automl_framework.py

# Run on your own data
python custom_example.py
```

### Use on Your Own CSV
```python
from automl_framework import AutoML
import pandas as pd

df = pd.read_csv('your_data.csv')
X  = df.drop('target_column', axis=1).values
y  = df['target_column'].values

automl = AutoML(task='classification')  # or 'regression'
automl.fit(X, y)
automl.report()
automl.save('best_model.pkl')
```

### Output
```
✅ AutoML Complete in ~4–8 seconds per dataset
🏆 Best Model printed with full leaderboard
📁 automl_results.json saved
📁 *_model.pkl saved
```

---

## 🛠️ Tech Stack

| Tool | Used For |
|------|----------|
| **Python 3.10+** | All projects |
| **scikit-learn** | ML models, TF-IDF, metrics, pipelines |
| **numpy** | Array operations, math |
| **pandas** | Data handling |
| **joblib** | Model saving/loading |

---

## 🚀 Quick Start — All Projects

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-projects.git
cd ml-projects

# Install all dependencies
pip install numpy pandas scikit-learn joblib

# Run any project
python sales_forecasting/sales_forecast.py
python genetic_algorithms/genetic_algorithm.py
python multilabel_text/multilabel_categorization.py
python automl_framework/automl_framework.py
```

---

## 📂 Repository Structure

```
ml-projects/
│
├── sales_forecasting/
│   ├── sales_forecast.py
│   ├── requirements.txt
│   └── README.md
│
├── genetic_algorithms/
│   ├── genetic_algorithm.py
│   ├── requirements.txt
│   └── README.md
│
├── multilabel_text/
│   ├── multilabel_categorization.py
│   ├── requirements.txt
│   └── README.md
│
├── automl_framework/
│   ├── automl_framework.py
│   ├── custom_example.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md          ← You are here
```

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
