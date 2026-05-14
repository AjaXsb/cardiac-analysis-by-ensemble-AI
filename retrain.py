"""Retrain all 6 models on cleveland_new.csv with current sklearn/xgboost.

Saves matching .joblib files in this directory so backend.py loads cleanly
under the installed library versions. Preprocessing mirrors backend.py's
expected feature schema.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(HERE, "..", "data", "cleveland_new.csv"))

EXPECTED_COLUMNS = [
    'age', 'sex', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
    'max_heart_rate', 'exercise_induced_angina', 'st_depression', 'num_major_vessels',
    'chest_pain_asymptomatic', 'chest_pain_atypical_angina', 'chest_pain_non-anginal_pain',
    'chest_pain_typical_angina', 'rest_ecg_STT_abnormality',
    'rest_ecg_left_ventricular_hypertrophy', 'rest_ecg_normal',
    'st_slope_type_downsloping', 'st_slope_type_flat', 'st_slope_type_upsloping',
    'thalassemia_fixed', 'thalassemia_normal', 'thalassemia_reversible',
]

CP_MAP = {0: 'asymptomatic', 1: 'atypical_angina', 2: 'non-anginal_pain', 3: 'typical_angina'}
ECG_MAP = {0: 'left_ventricular_hypertrophy', 1: 'normal', 2: 'STT_abnormality'}
SLOPE_MAP = {0: 'downsloping', 1: 'flat', 2: 'upsloping'}
THAL_MAP = {0: 'nothing', 1: 'fixed', 2: 'normal', 3: 'reversible'}


def load_and_preprocess_cleveland(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'cp': 'chest_pain', 'trestbps': 'resting_blood_pressure', 'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar', 'restecg': 'rest_ecg', 'thalach': 'max_heart_rate',
        'exang': 'exercise_induced_angina', 'oldpeak': 'st_depression',
        'slope': 'st_slope_type', 'ca': 'num_major_vessels', 'thal': 'thalassemia',
    })
    df['chest_pain'] = df['chest_pain'].map(CP_MAP)
    df['rest_ecg'] = df['rest_ecg'].map(ECG_MAP)
    df['st_slope_type'] = df['st_slope_type'].map(SLOPE_MAP)
    df['thalassemia'] = df['thalassemia'].map(THAL_MAP)

    y = df['target']
    x = df.drop(columns=['target'])
    x = pd.get_dummies(x, columns=['chest_pain', 'rest_ecg', 'st_slope_type', 'thalassemia'])

    # Drop categorical artifacts not in EXPECTED_COLUMNS (e.g., thalassemia_nothing)
    for col in EXPECTED_COLUMNS:
        if col not in x.columns:
            x[col] = 0
    x = x[EXPECTED_COLUMNS].astype(int)
    return x, y


def tune_xgb(x, y):
    grid = {'max_depth': [3, 4, 5], 'subsample': [0.6, 0.8, 1.0],
            'learning_rate': [0.01, 0.05, 0.1]}
    gs = GridSearchCV(XGBClassifier(eval_metric='logloss'), grid,
                      scoring='accuracy', cv=5, n_jobs=-1)
    gs.fit(x, y)
    return gs.best_estimator_, gs.best_score_


def tune_lr(x, y):
    grid = {'solver': ['liblinear'], 'penalty': ['l2'],
            'C': [100, 10, 1.0, 0.1, 0.01]}
    gs = GridSearchCV(LogisticRegression(max_iter=1000), grid,
                      scoring='accuracy', cv=5, n_jobs=-1)
    gs.fit(x, y)
    return gs.best_estimator_, gs.best_score_


def tune_knn(x, y):
    grid = {'n_neighbors': np.arange(1, 11),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]}
    gs = GridSearchCV(KNeighborsClassifier(), grid,
                      scoring=make_scorer(f1_score), cv=5, n_jobs=-1)
    gs.fit(x, y)
    return gs.best_estimator_, gs.best_score_


def tune_rf(x, y):
    grid = {'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50],
            'n_estimators': [25, 50, 100, 200]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                      grid, scoring='accuracy', cv=5, n_jobs=-1)
    gs.fit(x, y)
    return gs.best_estimator_, gs.best_score_


def build_weighted_ensemble(x, y, xgb, lr, knn, rf, s_xgb, s_lr, s_knn, s_rf):
    models = [('xgb', xgb, s_xgb), ('lr', lr, s_lr),
              ('knn', knn, s_knn), ('rf', rf, s_rf)]
    total = sum(s for _, _, s in models)
    weights = [s / total for _, _, s in models]
    ens = VotingClassifier(
        estimators=[(n, m) for n, m, _ in models],
        voting='soft', weights=weights,
    )
    ens.fit(x, y)
    return ens


def build_stacked_ensemble(x, y, xgb, lr, knn, rf):
    base = [('xgb', xgb), ('lr', lr), ('knn', knn), ('rf', rf)]
    ens = StackingClassifier(estimators=base,
                             final_estimator=LogisticRegression(max_iter=1000), cv=5)
    ens.fit(x, y)
    return ens


def evaluate_and_report(name, model, x_test, y_test):
    pred = model.predict(x_test)
    print(f"{name:>10}: acc={accuracy_score(y_test, pred):.4f} "
          f"f1={f1_score(y_test, pred):.4f}")


def main():
    print(f"Loading {CSV_PATH}")
    x, y = load_and_preprocess_cleveland(CSV_PATH)
    print(f"Shape: {x.shape}, target balance: {dict(y.value_counts())}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    print("Tuning XGB..."); xgb, s_xgb = tune_xgb(x_train, y_train)
    print("Tuning LR...");  lr,  s_lr  = tune_lr(x_train, y_train)
    print("Tuning KNN..."); knn, s_knn = tune_knn(x_train, y_train)
    print("Tuning RF...");  rf,  s_rf  = tune_rf(x_train, y_train)

    print("Building weighted ensemble...")
    weighted = build_weighted_ensemble(x_train, y_train, xgb, lr, knn, rf,
                                       s_xgb, s_lr, s_knn, s_rf)
    print("Building stacked ensemble...")
    stacked = build_stacked_ensemble(x_train, y_train, xgb, lr, knn, rf)

    print("\n--- Test scores ---")
    for name, m in [('xgb', xgb), ('lr', lr), ('knn', knn), ('rf', rf),
                    ('weighted', weighted), ('stacked', stacked)]:
        evaluate_and_report(name, m, x_test, y_test)

    print("\nSaving models...")
    joblib.dump(knn, os.path.join(HERE, 'knn_model.joblib'))
    joblib.dump(lr,  os.path.join(HERE, 'lr_model.joblib'))
    joblib.dump(rf,  os.path.join(HERE, 'rf_model.joblib'))
    joblib.dump(xgb, os.path.join(HERE, 'xgb_model.joblib'))
    joblib.dump(weighted, os.path.join(HERE, 'weighted_ensemble_model.joblib'))
    joblib.dump(stacked,  os.path.join(HERE, 'stacked_ensemble_model.joblib'))
    print("Done.")


if __name__ == '__main__':
    main()
