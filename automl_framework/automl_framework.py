"""
AutoML Framework Implementation using Machine Learning
======================================================
AutoML = Automatic Machine Learning
- Auto Data Preprocessing
- Auto Feature Engineering
- Auto Model Selection (10 models)
- Auto Hyperparameter Tuning (GridSearch)
- Auto Evaluation & Best Model Export

Supports: Classification & Regression tasks
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA

# Models — Classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Models — Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Tuning & Evaluation
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, StratifiedKFold, KFold)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score, mean_absolute_error,
                              classification_report)

# Datasets for demo
from sklearn.datasets import (load_iris, load_breast_cancer,
                               load_diabetes, load_wine, make_classification,
                               make_regression)

import joblib


# ════════════════════════════════════════════════════════════
# STEP 1 — AUTO DATA PROFILER
# ════════════════════════════════════════════════════════════

class DataProfiler:
    """Automatically profiles any dataset"""

    def profile(self, X, y, task='classification'):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols      = df.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_total = df.isnull().sum().sum()

        profile = {
            'n_samples':      len(df),
            'n_features':     df.shape[1],
            'numeric_cols':   numeric_cols,
            'cat_cols':       cat_cols,
            'missing_values': int(missing_total),
            'missing_pct':    round(missing_total / df.size * 100, 2),
            'task':           task,
        }

        if task == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            profile['n_classes']    = len(classes)
            profile['class_counts'] = dict(zip(classes.tolist(), counts.tolist()))
            imbalance = max(counts) / min(counts)
            profile['imbalanced']   = imbalance > 1.5
            profile['imbalance_ratio'] = round(float(imbalance), 2)
        else:
            profile['target_mean'] = round(float(np.mean(y)), 4)
            profile['target_std']  = round(float(np.std(y)), 4)
            profile['target_min']  = round(float(np.min(y)), 4)
            profile['target_max']  = round(float(np.max(y)), 4)

        return profile


# ════════════════════════════════════════════════════════════
# STEP 2 — AUTO PREPROCESSOR
# ════════════════════════════════════════════════════════════

class AutoPreprocessor:
    """
    Automatically builds preprocessing pipeline:
    - Numeric: Impute missing → Scale
    - Categorical: Impute missing → OneHot Encode
    """

    def __init__(self, scaler='standard'):
        self.scaler_type = scaler
        self.preprocessor = None
        self.numeric_cols  = []
        self.cat_cols      = []

    def fit_transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols     = df.select_dtypes(include=['object','category']).columns.tolist()

        scaler = StandardScaler() if self.scaler_type == 'standard' else MinMaxScaler()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  scaler)
        ])

        transformers = [('num', numeric_pipeline, self.numeric_cols)]

        if self.cat_cols:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipeline, self.cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        X_transformed = self.preprocessor.fit_transform(df)
        return X_transformed

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        return self.preprocessor.transform(df)


# ════════════════════════════════════════════════════════════
# STEP 3 — AUTO FEATURE SELECTOR
# ════════════════════════════════════════════════════════════

class AutoFeatureSelector:
    """
    Auto selects top K features using statistical tests
    - Classification: f_classif (ANOVA F-test)
    - Regression:     f_regression
    """

    def __init__(self, task='classification', k='auto'):
        self.task = task
        self.k    = k
        self.selector = None

    def fit_transform(self, X, y):
        n_features = X.shape[1]
        k = max(1, int(n_features * 0.8)) if self.k == 'auto' else min(self.k, n_features)

        score_func = f_classif if self.task == 'classification' else f_regression
        self.selector = SelectKBest(score_func=score_func, k=k)
        X_new = self.selector.fit_transform(X, y)
        return X_new, k

    def transform(self, X):
        return self.selector.transform(X)


# ════════════════════════════════════════════════════════════
# STEP 4 — AUTO MODEL SELECTOR
# ════════════════════════════════════════════════════════════

CLASSIFICATION_MODELS = {
    'Logistic Regression':      (LogisticRegression(max_iter=1000, random_state=42),
                                  {'C': [0.1, 1.0, 10.0]}),
    'Decision Tree':            (DecisionTreeClassifier(random_state=42),
                                  {'max_depth': [3, 5, 10, None]}),
    'Random Forest':            (RandomForestClassifier(random_state=42),
                                  {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
    'Gradient Boosting':        (GradientBoostingClassifier(random_state=42),
                                  {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}),
    'Extra Trees':              (ExtraTreesClassifier(random_state=42),
                                  {'n_estimators': [50, 100]}),
    'SVM':                      (SVC(probability=True, random_state=42),
                                  {'C': [0.1, 1.0], 'kernel': ['rbf', 'linear']}),
    'KNN':                      (KNeighborsClassifier(),
                                  {'n_neighbors': [3, 5, 7, 11]}),
    'Naive Bayes':              (GaussianNB(),
                                  {}),
    'Ridge Classifier':         (RidgeClassifier(),
                                  {'alpha': [0.1, 1.0, 10.0]}),
}

REGRESSION_MODELS = {
    'Linear Regression':        (LinearRegression(), {}),
    'Ridge':                    (Ridge(),
                                  {'alpha': [0.1, 1.0, 10.0, 100.0]}),
    'Lasso':                    (Lasso(max_iter=2000),
                                  {'alpha': [0.01, 0.1, 1.0]}),
    'ElasticNet':               (ElasticNet(max_iter=2000),
                                  {'alpha': [0.01, 0.1], 'l1_ratio': [0.3, 0.5, 0.7]}),
    'Decision Tree':            (DecisionTreeRegressor(random_state=42),
                                  {'max_depth': [3, 5, 10, None]}),
    'Random Forest':            (RandomForestRegressor(random_state=42),
                                  {'n_estimators': [50, 100], 'max_depth': [5, None]}),
    'Gradient Boosting':        (GradientBoostingRegressor(random_state=42),
                                  {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}),
    'Extra Trees':              (ExtraTreesRegressor(random_state=42),
                                  {'n_estimators': [50, 100]}),
    'SVR':                      (SVR(),
                                  {'C': [0.1, 1.0], 'kernel': ['rbf', 'linear']}),
    'KNN':                      (KNeighborsRegressor(),
                                  {'n_neighbors': [3, 5, 7]}),
}


class AutoModelSelector:
    """
    Trains ALL models, tunes hyperparameters, picks best one
    """

    def __init__(self, task='classification', cv=5, tune=True):
        self.task    = task
        self.cv      = cv
        self.tune    = tune
        self.results = {}
        self.best_model      = None
        self.best_model_name = None
        self.best_score      = float('-inf')

    def _score_metric(self):
        return 'accuracy' if self.task == 'classification' else 'r2'

    def fit(self, X_train, y_train):
        models = (CLASSIFICATION_MODELS if self.task == 'classification'
                  else REGRESSION_MODELS)

        cv_strategy = (StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
                       if self.task == 'classification'
                       else KFold(n_splits=self.cv, shuffle=True, random_state=42))

        print(f"\n{'Model':<25} {'CV Score':>10} {'Std':>8} {'Time':>8}")
        print("-" * 55)

        for name, (model, param_grid) in models.items():
            t_start = time.time()

            if self.tune and param_grid:
                gs = GridSearchCV(model, param_grid, cv=3,
                                  scoring=self._score_metric(),
                                  n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                best_estimator = gs.best_estimator_
                best_params    = gs.best_params_
            else:
                model.fit(X_train, y_train)
                best_estimator = model
                best_params    = {}

            # Cross-validation score
            cv_scores = cross_val_score(best_estimator, X_train, y_train,
                                        cv=cv_strategy,
                                        scoring=self._score_metric())
            mean_score = float(np.mean(cv_scores))
            std_score  = float(np.std(cv_scores))
            elapsed    = time.time() - t_start

            print(f"{name:<25} {mean_score:>10.4f} {std_score:>8.4f} {elapsed:>7.1f}s")

            self.results[name] = {
                'cv_score':   round(mean_score, 4),
                'cv_std':     round(std_score, 4),
                'best_params': best_params,
                'time_sec':   round(elapsed, 2),
            }

            if mean_score > self.best_score:
                self.best_score      = mean_score
                self.best_model      = best_estimator
                self.best_model_name = name

        print(f"\n🏆 Best Model : {self.best_model_name}  (CV Score = {self.best_score:.4f})")
        return self.best_model, self.best_model_name


# ════════════════════════════════════════════════════════════
# STEP 5 — AUTO EVALUATOR
# ════════════════════════════════════════════════════════════

class AutoEvaluator:

    def evaluate(self, model, X_test, y_test, task='classification'):
        y_pred = model.predict(X_test)
        metrics = {}

        if task == 'classification':
            metrics['accuracy']  = round(accuracy_score(y_test, y_pred), 4)
            metrics['f1_macro']  = round(f1_score(y_test, y_pred,
                                                   average='macro',
                                                   zero_division=0), 4)
            metrics['f1_weighted'] = round(f1_score(y_test, y_pred,
                                                     average='weighted',
                                                     zero_division=0), 4)
            # AUC for binary
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = round(roc_auc_score(y_test, proba), 4)

        else:
            metrics['r2']   = round(r2_score(y_test, y_pred), 4)
            metrics['mae']  = round(mean_absolute_error(y_test, y_pred), 4)
            metrics['rmse'] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            metrics['mape'] = round(
                float(np.mean(np.abs((y_test - y_pred) /
                                      np.where(y_test == 0, 1e-10, y_test)))) * 100, 2)

        return metrics, y_pred


# ════════════════════════════════════════════════════════════
# STEP 6 — AutoML MASTER CLASS
# ════════════════════════════════════════════════════════════

class AutoML:
    """
    One-stop AutoML class:
    automl = AutoML(task='classification')
    automl.fit(X, y)
    automl.predict(X_new)
    automl.report()
    """

    def __init__(self, task='classification', test_size=0.2,
                 cv=5, tune=True, scaler='standard'):
        self.task       = task
        self.test_size  = test_size
        self.cv         = cv
        self.tune       = tune
        self.scaler     = scaler

        self.profiler    = DataProfiler()
        self.preprocessor= AutoPreprocessor(scaler=scaler)
        self.feat_sel    = AutoFeatureSelector(task=task)
        self.selector    = AutoModelSelector(task=task, cv=cv, tune=tune)
        self.evaluator   = AutoEvaluator()

        self.profile_     = {}
        self.model_results= {}
        self.best_model   = None
        self.best_name    = None
        self.test_metrics = {}
        self._fitted      = False

    # ── fit ──────────────────────────────────────────────────
    def fit(self, X, y):
        print("\n" + "═"*60)
        print("   AutoML FRAMEWORK — AUTO TRAINING")
        print("═"*60)

        t_total = time.time()

        # 1. Profile
        print("\n📊 Step 1/5 — Data Profiling...")
        self.profile_ = self.profiler.profile(X, y, self.task)
        print(f"   Samples: {self.profile_['n_samples']}  |  "
              f"Features: {self.profile_['n_features']}  |  "
              f"Missing: {self.profile_['missing_pct']}%")

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42,
            stratify=y if self.task == 'classification' else None)
        self._X_test, self._y_test = X_test, y_test

        # 3. Preprocess
        print("\n⚙️  Step 2/5 — Auto Preprocessing...")
        X_train_p = self.preprocessor.fit_transform(
            pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train)
        X_test_p  = self.preprocessor.transform(
            pd.DataFrame(X_test)  if not isinstance(X_test,  pd.DataFrame) else X_test)
        print(f"   After preprocessing: {X_train_p.shape[1]} features")

        # 4. Feature Selection
        print("\n🔍 Step 3/5 — Auto Feature Selection...")
        X_train_s, k = self.feat_sel.fit_transform(X_train_p, y_train)
        X_test_s     = self.feat_sel.transform(X_test_p)
        print(f"   Selected top {k} / {X_train_p.shape[1]} features")

        # 5. Model Selection + Tuning
        print(f"\n🤖 Step 4/5 — Auto Model Selection & Tuning "
              f"({'GridSearchCV ON' if self.tune else 'No tuning'})...")
        self.best_model, self.best_name = self.selector.fit(X_train_s, y_train)
        self.model_results = self.selector.results

        # 6. Final Evaluation
        print("\n📈 Step 5/5 — Final Evaluation on Test Set...")
        self.test_metrics, y_pred = self.evaluator.evaluate(
            self.best_model, X_test_s, y_test, self.task)

        self._X_test_s = X_test_s
        self._fitted   = True
        total_time     = time.time() - t_total

        print(f"\n{'─'*60}")
        print(f"✅  AutoML Complete in {total_time:.1f}s")
        print(f"🏆  Best Model : {self.best_name}")
        for k_m, v_m in self.test_metrics.items():
            print(f"    {k_m:<15}: {v_m}")
        print(f"{'─'*60}\n")

        return self

    # ── predict ──────────────────────────────────────────────
    def predict(self, X_new):
        assert self._fitted, "Call fit() first"
        X_df = pd.DataFrame(X_new) if not isinstance(X_new, pd.DataFrame) else X_new
        X_p  = self.preprocessor.transform(X_df)
        X_s  = self.feat_sel.transform(X_p)
        return self.best_model.predict(X_s)

    # ── report ───────────────────────────────────────────────
    def report(self):
        report = {
            'task':           self.task,
            'data_profile':   self.profile_,
            'best_model':     self.best_name,
            'test_metrics':   self.test_metrics,
            'all_models':     self.model_results,
            'leaderboard':    sorted(
                [(n, v['cv_score']) for n, v in self.model_results.items()],
                key=lambda x: x[1], reverse=True
            )
        }
        return report

    # ── save ─────────────────────────────────────────────────
    def save(self, path='automl_model.pkl'):
        joblib.dump({'model': self.best_model,
                     'preprocessor': self.preprocessor,
                     'feat_sel': self.feat_sel,
                     'name': self.best_name}, path)
        print(f"💾 Model saved → {path}")


# ════════════════════════════════════════════════════════════
# DEMO — Run on 3 datasets
# ════════════════════════════════════════════════════════════

def run_demo():
    all_results = {}

    demos = [
        {
            'name':  'Iris Flower Classification',
            'task':  'classification',
            'load':  load_iris,
        },
        {
            'name':  'Breast Cancer Detection',
            'task':  'classification',
            'load':  load_breast_cancer,
        },
        {
            'name':  'Diabetes Regression',
            'task':  'regression',
            'load':  load_diabetes,
        },
    ]

    for demo in demos:
        print(f"\n\n{'★'*60}")
        print(f"  DATASET: {demo['name']}")
        print(f"{'★'*60}")

        data   = demo['load']()
        X, y   = data.data, data.target

        automl = AutoML(task=demo['task'], cv=5, tune=True)
        automl.fit(X, y)
        report = automl.report()

        # Save model
        safe_name = demo['name'].replace(' ', '_').lower()
        automl.save(f"{safe_name}_model.pkl")

        all_results[demo['name']] = report

        # Print leaderboard
        print(f"\n📊 Leaderboard — {demo['name']}")
        print(f"{'Rank':<6}{'Model':<25}{'CV Score':>10}")
        print("-"*43)
        for rank, (name, score) in enumerate(report['leaderboard'], 1):
            marker = " ← best" if rank == 1 else ""
            print(f"{rank:<6}{name:<25}{score:>10.4f}{marker}")

    # Save full report
    with open('automl_results.json', 'w') as f:
        import json as _json
        class NumpyEncoder(_json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, (np.bool_,)): return bool(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print("\n\n📁 Full results saved → automl_results.json")

    return all_results


if __name__ == '__main__':
    run_demo()
