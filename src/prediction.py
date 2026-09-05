"""
Career Prediction & Inference Module
Loads trained model artifacts, preprocesses student inputs, and generates
primary career recommendations with calibrated confidence scores and alternative roles.
"""

import os
import joblib
import numpy as np
import pandas as pd
import sys
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data_preprocessing import add_engineered_features, FEATURE_COLUMNS
except ModuleNotFoundError:
    from data_preprocessing import add_engineered_features, FEATURE_COLUMNS

class CareerPredictor:
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "best_career_model.joblib")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(base_dir, "data", "processed", "preprocessor.joblib")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train_model.py first.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run data_preprocessing.py first.")
            
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.model_name = self.artifact.get("model_name", "Classifier")
        self.classes = self.artifact["classes"]
        
        self.preprocessor = joblib.load(preprocessor_path)
        self.scaler = self.preprocessor["scaler"]
        self.label_encoder = self.preprocessor["label_encoder"]
        self.feature_columns = self.preprocessor["feature_columns"]

    def prepare_features(self, profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert student profile dictionary into properly engineered and scaled DataFrame.
        """
        # Create single-row DataFrame
        df_raw = pd.DataFrame([profile])
        
        # Ensure all required raw columns exist
        raw_cols = [
            "python_score", "sql_score", "excel_score", "stats_score",
            "ml_score", "powerbi_score", "comm_score", "aptitude_score",
            "cgpa", "projects_count", "internships_count", "certifications_count"
        ]
        for col in raw_cols:
            if col not in df_raw.columns:
                df_raw[col] = 0.0
                
        # Feature engineering
        df_enriched = add_engineered_features(df_raw)
        
        # Select exact feature set in order
        X_df = df_enriched[self.feature_columns].copy()
        
        return X_df

    def predict(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate career recommendation with probabilities and alternative roles.
        """
        X_df = self.prepare_features(profile)
        X_scaled = self.scaler.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Check if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled_df)[0]
        else:
            probs = np.zeros(len(self.classes))
            pred_idx = self.model.predict(X_scaled_df)[0]
            probs[pred_idx] = 1.0

        # Sort roles by descending probability
        role_probs = []
        for idx, prob in enumerate(probs):
            role_name = self.classes[idx]
            role_probs.append({
                "role": role_name,
                "probability": round(float(prob) * 100, 2)
            })
            
        role_probs.sort(key=lambda x: x["probability"], reverse=True)
        
        primary_role = role_probs[0]["role"]
        confidence = role_probs[0]["probability"]
        alternative_roles = role_probs[1:3]
        
        return {
            "model_used": self.model_name,
            "primary_recommendation": primary_role,
            "confidence_score": confidence,
            "alternative_roles": alternative_roles,
            "all_role_probabilities": role_probs,
            "engineered_features": X_df.iloc[0].to_dict(),
            "scaled_features": X_scaled[0]
        }

def get_career_prediction(profile: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    """Helper functional interface for quick inference."""
    predictor = CareerPredictor(model_path=model_path)
    return predictor.predict(profile)

if __name__ == "__main__":
    sample_student = {
        "python_score": 88,
        "sql_score": 75,
        "excel_score": 60,
        "stats_score": 85,
        "ml_score": 84,
        "powerbi_score": 55,
        "comm_score": 70,
        "aptitude_score": 78,
        "cgpa": 8.5,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2
    }
    pred = get_career_prediction(sample_student)
    print("Inference Result:")
    print(f"Recommended Role: {pred['primary_recommendation']} ({pred['confidence_score']}%)")
    print("Alternative Roles:", pred['alternative_roles'])
