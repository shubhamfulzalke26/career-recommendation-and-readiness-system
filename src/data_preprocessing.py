"""
Data Preprocessing & Feature Engineering Module
Performs data cleaning, leak-free feature transformations, train-test splitting,
and artifact persistence for the ML pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Base features used for model input
FEATURE_COLUMNS = [
    "python_score",
    "sql_score",
    "excel_score",
    "stats_score",
    "ml_score",
    "powerbi_score",
    "comm_score",
    "aptitude_score",
    "cgpa",
    "projects_count",
    "internships_count",
    "certifications_count",
    # Engineered features
    "core_tech_avg",
    "analytics_avg",
    "business_comm_avg",
    "practical_score"
]

TARGET_COLUMN = "target_role"

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific composite features based on student competence.
    """
    df_feat = df.copy()
    
    # 1. Technical domain average
    df_feat["core_tech_avg"] = df_feat[["python_score", "sql_score", "stats_score", "ml_score"]].mean(axis=1)
    
    # 2. Business analytics domain average
    df_feat["analytics_avg"] = df_feat[["sql_score", "excel_score", "powerbi_score"]].mean(axis=1)
    
    # 3. Communication & cognitive domain average
    df_feat["business_comm_avg"] = df_feat[["comm_score", "aptitude_score"]].mean(axis=1)
    
    # 4. Practical experiential exposure index (projects, internships, certifications)
    df_feat["practical_score"] = (
        (df_feat["projects_count"] * 5.0) +
        (df_feat["internships_count"] * 10.0) +
        (df_feat["certifications_count"] * 5.0)
    )
    
    return df_feat

def preprocess_pipeline(
    raw_data_path: str = None,
    output_dir: str = None,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    End-to-end data processing workflow ensuring zero data leakage.
    Fits scaler exclusively on the training partition.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_data_path is None:
        raw_data_path = os.path.join(base_dir, "data", "raw", "student_placement_synthetic.csv")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "processed")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load raw data
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset from {raw_data_path} with shape {df.shape}")
    
    # 2. Check and handle missing values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Warning: Found {null_counts} null values. Imputing...")
        df = df.dropna()
    else:
        print("[OK] Zero missing values detected.")
        
    # 3. Add engineered features
    df_enriched = add_engineered_features(df)
    
    # 4. Separate features and target
    X = df_enriched[FEATURE_COLUMNS].copy()
    y = df_enriched[TARGET_COLUMN].copy()
    
    # Keep metadata & demographics separate for fairness auditing
    audit_columns = ["student_id", "gender", "college_tier", "socioeconomic_status", "placement_status"]
    df_audit = df_enriched[audit_columns].copy()
    
    # 5. Label Encode target classes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 6. Stratified Train-Test Split to maintain class proportions
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_encoded, df.index, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # 7. Fit Scaler ONLY on Training Split (strictly avoids data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrames for saved features
    df_X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
    df_X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)
    
    df_y_train = pd.Series(y_train, name=TARGET_COLUMN, index=X_train.index)
    df_y_test = pd.Series(y_test, name=TARGET_COLUMN, index=X_test.index)
    
    # Separate audit sets
    audit_train = df_audit.loc[idx_train]
    audit_test = df_audit.loc[idx_test]
    
    # 8. Save artifacts
    df_X_train.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    df_X_test.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
    df_y_train.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)
    df_y_test.to_csv(os.path.join(output_dir, "test_target.csv"), index=False)
    audit_train.to_csv(os.path.join(output_dir, "audit_train.csv"), index=False)
    audit_test.to_csv(os.path.join(output_dir, "audit_test.csv"), index=False)
    
    # Save preprocessing metadata object
    preprocessor = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(label_encoder.classes_),
        "raw_features": [c for c in FEATURE_COLUMNS if c not in ["core_tech_avg", "analytics_avg", "business_comm_avg", "practical_score"]]
    }
    
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[OK] Preprocessor saved to {preprocessor_path}")
    print(f"Training split: {len(X_train)} samples, Test split: {len(X_test)} samples")
    print(f"Target classes: {list(label_encoder.classes_)}")
    
    return preprocessor

if __name__ == "__main__":
    preprocess_pipeline()
