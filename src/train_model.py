"""
Model Training & Evaluation Module
Trains, compares, evaluates, and exports multiple multi-class classifiers
for student career role recommendation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def train_and_compare_models(processed_dir: str = None, models_dir: str = None):
    """
    Train 4 candidate classifiers on processed training data, evaluate on test data,
    record comprehensive performance metrics, and persist the top-performing model.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if processed_dir is None:
        processed_dir = os.path.join(base_dir, "data", "processed")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
        
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load preprocessed features and targets
    X_train = pd.read_csv(os.path.join(processed_dir, "train_features.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "test_features.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "train_target.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "test_target.csv")).squeeze()
    
    preprocessor = joblib.load(os.path.join(processed_dir, "preprocessor.joblib"))
    class_names = preprocessor["classes"]
    
    # 2. Define candidate models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
    }
    
    comparison_results = {}
    fitted_models = {}
    best_model_name = None
    best_f1 = -1.0
    
    print("\n" + "="*60)
    print("      AIML006 MODEL COMPARISON & EVALUATION")
    print("="*60)
    
    for name, model in candidates.items():
        # Train model
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        p_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        r_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        comparison_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p_macro), 4),
            "precision_weighted": round(float(p_weighted), 4),
            "recall_macro": round(float(r_macro), 4),
            "recall_weighted": round(float(r_weighted), 4),
            "f1_macro": round(float(f1_m), 4),
            "f1_weighted": round(float(f1_w), 4),
            "confusion_matrix": cm
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:         {acc*100:.2f}%")
        print(f"  Macro F1-Score:   {f1_m*100:.2f}%")
        print(f"  Weighted F1:      {f1_w*100:.2f}%")
        
        if f1_m > best_f1:
            best_f1 = f1_m
            best_model_name = name

    print("\n" + "-"*60)
    print(f"Winning Model: {best_model_name} (Macro F1 = {best_f1*100:.2f}%)")
    print("-"*60)
    
    # 3. Save comparison results
    results_path = os.path.join(models_dir, "model_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"[OK] Saved comparison results to: {results_path}")
    
    # 4. Save best model and all candidates
    best_model = fitted_models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_career_model.joblib")
    
    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "classes": class_names,
        "feature_names": preprocessor["feature_columns"],
        "metrics": comparison_results[best_model_name]
    }
    joblib.dump(model_artifact, best_model_path)
    print(f"[OK] Saved best model artifact to: {best_model_path}")
    
    # Save all models for interactive switching in Streamlit
    all_models_path = os.path.join(models_dir, "all_trained_models.joblib")
    joblib.dump({k: {"model": m, "metrics": comparison_results[k]} for k, m in fitted_models.items()}, all_models_path)
    
    # 5. Plot and save Confusion Matrix for the winning model
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix: {best_model_name} (Accuracy: {comparison_results[best_model_name]['accuracy']*100:.1f}%)")
    plt.xlabel("Predicted Career Role")
    plt.ylabel("Actual Career Role")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Confusion Matrix heatmap to: {cm_path}")
    
    return comparison_results, best_model_name

if __name__ == "__main__":
    train_and_compare_models()
