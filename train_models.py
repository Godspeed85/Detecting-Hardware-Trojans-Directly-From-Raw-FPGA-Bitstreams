#!/usr/bin/env python3
"""
FINAL MERGED TRAINING PIPELINE
Combines:
✔ PCA-based clean academic pipeline
✔ Multi-model supervised learning (RF, GBM, SVM)
✔ Anomaly detection (IsolationForest, OneClassSVM)
✔ Auto model selection based on F1
✔ Automatic plots for presentation
✔ Saves best model + scaler + PCA for reproducibility

USAGE:
    python train_models.py features.csv --pca
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, roc_auc_score)

import joblib


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(df.head())
    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # ADD 'file' and 'path' to the exclusion list to ensure they are not treated as features
    feature_cols = [c for c in df.columns if c not in 
                    ['file', 'path', 'filename', 'design_name', 'label', 'trojan_type']]
    
    X = df[feature_cols].values
    y = (df['label'] == 'malicious').astype(int).values

    # This line correctly handles any missing or non-finite *numerical* values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, df, feature_cols


# ============================================================
# 2. PCA (OPTIONAL)
# ============================================================

def apply_pca_if_needed(X_train, X_test, enable_pca, n_components=50):
    if not enable_pca:
        return X_train, X_test, None

    print("\nApplying PCA...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA reduced from {X_train.shape[1]} → {n_components} dims")

    return X_train_pca, X_test_pca, pca


# ============================================================
# 3. SUPERVISED MODELS
# ============================================================

def train_supervised_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    best_f1 = -1
    best_model_name = None

    print("\n" + "="*70)
    print("SUPERVISED LEARNING")
    print("="*70)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nBEST SUPERVISED MODEL → {best_model_name} (F1 = {best_f1:.3f})")
    return results, best_model_name


# ============================================================
# 4. ANOMALY DETECTION MODELS
# ============================================================

def train_anomaly_models(X_train, X_test, y_test):
    models = {
        "Isolation Forest": IsolationForest(contamination=0.30, random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.30, kernel='rbf')
    }

    results = {}

    print("\n" + "="*70)
    print("ANOMALY DETECTION")
    print("="*70)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train)

        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0)

        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return results


# ============================================================
# 5. PLOTS
# ============================================================

def generate_plots(results, best_model_name, y_test, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    print(f"Saved: {output_dir}/confusion_matrix.png")

    # ROC Curve
    if "y_proba" in results[best_model_name]:
        y_proba = results[best_model_name]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1], [0,1], "k--")
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(f"{output_dir}/roc_curve.png")
        print(f"Saved: {output_dir}/roc_curve.png")

    # Model Comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    f1_scores = [results[n]["f1"] for n in names]

    plt.bar(names, f1_scores)
    plt.xticks(rotation=20)
    plt.title("Model F1 Comparison")
    plt.savefig(f"{output_dir}/model_comparison.png")
    print(f"Saved: {output_dir}/model_comparison.png")


# ============================================================
# 6. SAVE BEST MODEL BUNDLE
# ============================================================

def save_bundle(best_model, scaler, pca, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(best_model, f"{output_dir}/model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    if pca:
        joblib.dump(pca, f"{output_dir}/pca.pkl")

    print("\nSaved model bundle:")
    print(f"  {output_dir}/model.pkl")
    print(f"  {output_dir}/scaler.pkl")
    if pca:
        print(f"  {output_dir}/pca.pkl")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to features.csv")
    parser.add_argument("--pca", action="store_true", help="Enable PCA")
    args = parser.parse_args()

    X, y, df, feature_cols = load_data(args.csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_test, pca = apply_pca_if_needed(X_train, X_test, args.pca)

    supervised, best_supervised_name = train_supervised_models(
        X_train, X_test, y_train, y_test
    )

    anomaly = train_anomaly_models(X_train, X_test, y_test)

    all_results = {**supervised, **anomaly}

    generate_plots(all_results, best_supervised_name, y_test)

    best_model = supervised[best_supervised_name]["model"]
    save_bundle(best_model, scaler, pca)

    print("\nDONE — Ready for your paper + presentation!")

if __name__ == "__main__":
    main()
