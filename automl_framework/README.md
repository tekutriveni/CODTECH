# AutoML Framework using Machine Learning        
 Intern ID:CITS1643
Automatically selects the best ML model for any dataset — no manual tuning needed!

## What AutoML Does Automatically

```
Your Data
   ↓
📊 Data Profiling      — shape, missing values, class balance
   ↓
⚙️  Preprocessing       — impute missing, scale, encode categories
   ↓
🔍 Feature Selection   — picks top 80% important features
   ↓
🤖 Model Selection     — trains 9–10 models, compares all
   ↓
🎯 Hyperparameter Tune — GridSearchCV for best params
   ↓
📈 Evaluation          — accuracy / R², F1, RMSE, etc.
   ↓
💾 Best Model Saved    — .pkl file ready to use
```

## Models Tried Automatically

### Classification (9 models)
- Logistic Regression, Decision Tree, Random Forest
- Gradient Boosting, Extra Trees, SVM
- KNN, Naive Bayes, Ridge Classifier

### Regression (10 models)
- Linear Regression, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, Gradient Boosting
- Extra Trees, SVR, KNN

## How to Run

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Run demo (3 datasets)
```bash
python automl_framework.py
```

### Step 3 — Run on custom data
```bash
python custom_example.py
```

### Step 4 — Use on YOUR CSV
```python
from automl_framework import AutoML
import pandas as pd

df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

automl = AutoML(task='classification')  # or 'regression'
automl.fit(X, y)
automl.report()
automl.save('my_model.pkl')
```

## Output Files
- `automl_results.json` — full leaderboard & metrics
- `*_model.pkl` — saved best model

## File Structure
```
automl_framework/
├── automl_framework.py   ← Main AutoML class
├── custom_example.py     ← Use on your own data
├── requirements.txt
├── README.md
└── automl_results.json   ← Generated after run
```
