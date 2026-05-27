"""
AutoML on Custom Dataset Example
=================================
ఇక్కడ మీ own CSV data మీద AutoML run చేయవచ్చు.
"""

import pandas as pd
import numpy as np
from automl_framework import AutoML

# ─── Option 1: CSV file load చేయి ────────────────────────────
# df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1).values
# y = df['target_column'].values

# ─── Option 2: Demo synthetic data ───────────────────────────
from sklearn.datasets import make_classification, make_regression

print("=" * 50)
print("Example 1: Custom Classification Dataset")
print("=" * 50)
X_cls, y_cls = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

automl_cls = AutoML(task='classification', cv=5, tune=True)
automl_cls.fit(X_cls, y_cls)
report_cls = automl_cls.report()

print("\nTop 3 Models:")
for rank, (name, score) in enumerate(report_cls['leaderboard'][:3], 1):
    print(f"  {rank}. {name}: {score:.4f}")

# ─── Predict on new samples ──────────────────────────────────
X_new = X_cls[:5]
preds = automl_cls.predict(X_new)
print(f"\nPredictions on 5 new samples: {preds}")

print("\n" + "=" * 50)
print("Example 2: Custom Regression Dataset")
print("=" * 50)
X_reg, y_reg = make_regression(
    n_samples=300,
    n_features=15,
    n_informative=8,
    noise=0.1,
    random_state=42
)

automl_reg = AutoML(task='regression', cv=5, tune=True)
automl_reg.fit(X_reg, y_reg)
report_reg = automl_reg.report()

print("\nTop 3 Models:")
for rank, (name, score) in enumerate(report_reg['leaderboard'][:3], 1):
    print(f"  {rank}. {name}: {score:.4f}")

# Save models
automl_cls.save('custom_classification_model.pkl')
automl_reg.save('custom_regression_model.pkl')

print("\n✅ Custom examples done!")
