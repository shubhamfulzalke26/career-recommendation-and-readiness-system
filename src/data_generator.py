"""
Data Generator Module for Student Placement & Career Recommendation System.
Generates realistic, synthetic student profiles with track-aligned skills,
academic metrics, career roles, and demographic attributes for fairness auditing.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset of student profiles with realistic specialization
    tracks, cross-skill correlations, and target career roles.
    """
    np.random.seed(random_seed)
    
    roles = [
        "Data Analyst",
        "Business Analyst",
        "Data Scientist",
        "ML Intern",
        "BI Analyst",
        "Data Science Intern"
    ]
    
    # Assign target career role tracks with realistic college cohort distribution
    role_probs = [0.20, 0.22, 0.18, 0.14, 0.14, 0.12]
    assigned_roles = np.random.choice(roles, size=num_samples, p=role_probs)
    
    # Base latent general ability for each student
    general_ability = np.random.normal(0, 1.0, num_samples)
    
    # Track-specific skill baselines (mean score profiles for each role)
    # [python, sql, excel, stats, ml, powerbi, comm, aptitude]
    role_baselines = {
        "Data Analyst":        [62, 82, 80, 68, 52, 78, 68, 70],
        "Business Analyst":    [50, 62, 84, 60, 45, 72, 85, 80],
        "Data Scientist":      [85, 76, 60, 84, 82, 58, 66, 76],
        "ML Intern":           [86, 68, 54, 78, 86, 50, 62, 74],
        "BI Analyst":          [56, 80, 78, 62, 48, 86, 74, 70],
        "Data Science Intern": [74, 72, 65, 75, 70, 62, 68, 72]
    }
    
    python_scores = []
    sql_scores = []
    excel_scores = []
    stats_scores = []
    ml_scores = []
    powerbi_scores = []
    comm_scores = []
    aptitude_scores = []
    
    for i in range(num_samples):
        role = assigned_roles[i]
        base = role_baselines[role]
        ga = general_ability[i]
        
        # Individual variation with correlated ability
        indiv_noise = np.random.normal(0, 7.5, 8)
        scores = np.array(base) + (ga * 5.0) + indiv_noise
        scores = np.clip(np.round(scores), 25, 99).astype(int)
        
        python_scores.append(scores[0])
        sql_scores.append(scores[1])
        excel_scores.append(scores[2])
        stats_scores.append(scores[3])
        ml_scores.append(scores[4])
        powerbi_scores.append(scores[5])
        comm_scores.append(scores[6])
        aptitude_scores.append(scores[7])
        
    python_score = np.array(python_scores)
    sql_score = np.array(sql_scores)
    excel_score = np.array(excel_scores)
    stats_score = np.array(stats_scores)
    ml_score = np.array(ml_scores)
    powerbi_score = np.array(powerbi_scores)
    comm_score = np.array(comm_scores)
    aptitude_score = np.array(aptitude_scores)
    
    # Academic & Experiential metrics with track affinity
    cgpa_base = 7.1 + (general_ability * 0.6) + np.random.normal(0, 0.35, num_samples)
    cgpa = np.clip(np.round(cgpa_base, 2), 5.5, 9.8)
    
    projects_list = []
    internships_list = []
    certifications_list = []
    
    for i in range(num_samples):
        role = assigned_roles[i]
        ga = general_ability[i]
        
        # ML Intern and Data Scientist students tend to have more technical projects
        proj_bias = 1.0 if role in ["ML Intern", "Data Scientist"] else 0.2
        proj = np.clip(np.round(2.2 + proj_bias + (ga * 0.6) + np.random.normal(0, 0.6)), 0, 6)
        projects_list.append(int(proj))
        
        # Internships (0 to 3)
        intern_bias = 0.4 if role in ["Data Scientist", "Data Analyst", "BI Analyst"] else 0.2
        intern_prob = np.clip(0.3 + intern_bias + (ga * 0.1), 0.05, 0.85)
        intern = np.random.binomial(3, intern_prob / 3)
        internships_list.append(int(intern))
        
        # Certifications (0 to 5)
        cert_bias = 1.2 if role in ["BI Analyst", "Data Analyst", "Data Science Intern"] else 0.5
        cert = np.clip(np.round(1.5 + cert_bias + np.random.normal(0, 0.7)), 0, 5)
        certifications_list.append(int(cert))
        
    projects_count = np.array(projects_list)
    internships_count = np.array(internships_list)
    certifications_count = np.array(certifications_list)
    
    # Demographic attributes (Auditing ONLY - strictly excluded from model training)
    genders = np.random.choice(["Male", "Female", "Other"], size=num_samples, p=[0.52, 0.45, 0.03])
    college_tiers = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=num_samples, p=[0.25, 0.45, 0.30])
    socioeconomic_status = np.random.choice(["Urban", "Semi-Urban", "Rural"], size=num_samples, p=[0.45, 0.35, 0.20])
    
    # Placement outcome calculation (Realistic ~62% placement rate)
    placement_index = (
        0.18 * (python_score + sql_score) / 2 +
        0.15 * (stats_score + ml_score) / 2 +
        0.15 * (comm_score + aptitude_score) / 2 +
        0.22 * (cgpa * 10) +
        0.15 * (projects_count * 15) +
        0.15 * (internships_count * 25)
    )
    tier_boost = {"Tier 1": 4.0, "Tier 2": 0.0, "Tier 3": -4.0}
    adjusted_index = placement_index + np.array([tier_boost[t] for t in college_tiers]) + np.random.normal(0, 4.0, num_samples)
    placement_status = np.where(adjusted_index >= 56.0, "Placed", "Not Placed")
    
    student_ids = [f"STU_{i+1:04d}" for i in range(num_samples)]
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "python_score": python_score,
        "sql_score": sql_score,
        "excel_score": excel_score,
        "stats_score": stats_score,
        "ml_score": ml_score,
        "powerbi_score": powerbi_score,
        "comm_score": comm_score,
        "aptitude_score": aptitude_score,
        "cgpa": cgpa,
        "projects_count": projects_count,
        "internships_count": internships_count,
        "certifications_count": certifications_count,
        "target_role": assigned_roles,
        "placement_status": placement_status,
        "gender": genders,
        "college_tier": college_tiers,
        "socioeconomic_status": socioeconomic_status
    })
    
    return df

def save_synthetic_dataset(output_path: str = None, num_samples: int = 1500) -> str:
    """Generate and save the synthetic dataset to CSV."""
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "raw", "student_placement_synthetic.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_data(num_samples=num_samples)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} synthetic student profiles saved to: {output_path}")
    print("\nRole distribution:")
    print(df['target_role'].value_counts())
    print("\nPlacement status:")
    print(df['placement_status'].value_counts())
    return output_path

if __name__ == "__main__":
    save_synthetic_dataset()
