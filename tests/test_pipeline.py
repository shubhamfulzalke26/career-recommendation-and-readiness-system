"""
Comprehensive Test Suite for AIML006
Tests data generation, preprocessing, model inference, readiness scoring,
skill assessments, skill gap analysis, SHAP explainability, and fairness audits.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure root directory is on python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_generator import generate_synthetic_data
from src.data_preprocessing import add_engineered_features, FEATURE_COLUMNS
from src.readiness_scorer import calculate_placement_readiness
from src.skill_assessment import load_questions, get_available_skills, evaluate_assessment
from src.prediction import CareerPredictor
from src.skill_gap import analyze_skill_gaps, ROLE_BENCHMARKS
from src.recommendation import generate_personalized_recommendations
from src.explainability import ModelExplainer
from src.fairness import compute_demographic_parity, audit_career_model_fairness

@pytest.fixture
def sample_student():
    return {
        "python_score": 85,
        "sql_score": 80,
        "excel_score": 70,
        "stats_score": 82,
        "ml_score": 84,
        "powerbi_score": 60,
        "comm_score": 72,
        "aptitude_score": 78,
        "cgpa": 8.4,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2
    }

def test_synthetic_data_generation():
    df = generate_synthetic_data(num_samples=100, random_seed=42)
    assert len(df) == 100
    assert df.isnull().sum().sum() == 0
    assert "target_role" in df.columns
    assert "placement_status" in df.columns
    assert "gender" in df.columns
    assert set(df["target_role"].unique()).issubset({
        "Data Analyst", "Business Analyst", "Data Scientist",
        "ML Intern", "BI Analyst", "Data Science Intern"
    })

def test_feature_engineering():
    raw_df = pd.DataFrame([{
        "python_score": 80, "sql_score": 70, "excel_score": 60, "stats_score": 90,
        "ml_score": 85, "powerbi_score": 65, "comm_score": 75, "aptitude_score": 80,
        "cgpa": 8.0, "projects_count": 2, "internships_count": 1, "certifications_count": 1
    }])
    enriched = add_engineered_features(raw_df)
    assert "core_tech_avg" in enriched.columns
    assert "analytics_avg" in enriched.columns
    assert "business_comm_avg" in enriched.columns
    assert "practical_score" in enriched.columns
    assert enriched["core_tech_avg"].iloc[0] == (80 + 70 + 90 + 85) / 4

def test_readiness_scoring(sample_student):
    result = calculate_placement_readiness(sample_student)
    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["readiness_tier"] in ["Beginner", "Developing", "Placement Ready", "Highly Ready"]
    assert "breakdown" in result
    assert "technical_competence" in result["breakdown"]

def test_skill_assessment_bank():
    questions = load_questions()
    assert len(questions) == 160
    skills = get_available_skills()
    assert len(skills) == 8
    
    # Test evaluation
    sample_answers = {
        questions[0]["id"]: questions[0]["correct_answer"],
        questions[1]["id"]: "WRONG_ANSWER"
    }
    report = evaluate_assessment(sample_answers)
    assert report["total_questions"] == 2
    assert report["total_correct"] == 1
    assert report["overall_percentage"] == 50.0

def test_career_prediction(sample_student):
    predictor = CareerPredictor()
    pred = predictor.predict(sample_student)
    assert "primary_recommendation" in pred
    assert pred["primary_recommendation"] in predictor.classes
    assert 0.0 <= pred["confidence_score"] <= 100.0
    assert len(pred["alternative_roles"]) == 2
    assert len(pred["all_role_probabilities"]) == 6

def test_skill_gap_analysis(sample_student):
    gap_result = analyze_skill_gaps(sample_student, "Data Scientist")
    assert gap_result["target_role"] == "Data Scientist"
    assert len(gap_result["comparisons"]) == 8
    assert "priority_gaps" in gap_result
    assert "strong_skills" in gap_result

def test_personalized_recommendations(sample_student):
    gap_result = analyze_skill_gaps(sample_student, "Data Scientist")
    recs = generate_personalized_recommendations(gap_result)
    assert recs["target_role"] == "Data Scientist"
    assert "weekly_plan" in recs
    assert len(recs["weekly_plan"]) == 4
    assert "project_blueprint" in recs
    assert "title" in recs["project_blueprint"]

def test_shap_explainability(sample_student):
    explainer = ModelExplainer()
    explanation = explainer.explain_student_profile(sample_student)
    assert "recommended_role" in explanation
    assert len(explanation["attributions"]) == len(FEATURE_COLUMNS)
    assert "narrative" in explanation
    assert len(explanation["narrative"]) > 10

def test_fairness_audit():
    audit = audit_career_model_fairness()
    assert "gender_audit" in audit
    assert "college_tier_audit" in audit
    assert "socioeconomic_audit" in audit
    assert 0.0 <= audit["gender_audit"]["disparate_impact_ratio"] <= 1.5
    assert "ethical_disclosure" in audit
