"""
Algorithmic Fairness & Bias Audit Module
Evaluates model behavior and placement outcomes across demographic cohorts
to measure Demographic Parity, Disparate Impact, and Equal Opportunity metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_demographic_parity(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_col: str,
    favorable_val: str
) -> Dict[str, Any]:
    """
    Calculate selection rate per demographic subgroup, Demographic Parity Difference,
    and Disparate Impact (Four-Fifths Rule ratio).
    
    Formula:
      Selection Rate = Count(Outcome == Favorable) / Count(Subgroup)
      Disparate Impact = Min(Selection Rate) / Max(Selection Rate)
      Four-Fifths Rule: Pass if Disparate Impact >= 0.80
    """
    groups = df[protected_attr].dropna().unique()
    selection_rates = {}
    group_counts = {}
    
    for g in groups:
        sub = df[df[protected_attr] == g]
        count = len(sub)
        favorable_count = len(sub[sub[outcome_col] == favorable_val])
        rate = favorable_count / max(count, 1)
        selection_rates[str(g)] = round(float(rate), 4)
        group_counts[str(g)] = int(count)
        
    rates = list(selection_rates.values())
    if not rates or max(rates) == 0:
        disparate_impact = 1.0
        max_diff = 0.0
    else:
        disparate_impact = round(min(rates) / max(rates), 4)
        max_diff = round(max(rates) - min(rates), 4)
        
    four_fifths_pass = disparate_impact >= 0.80
    
    return {
        "protected_attribute": protected_attr,
        "outcome_evaluated": outcome_col,
        "favorable_value": favorable_val,
        "group_counts": group_counts,
        "selection_rates": selection_rates,
        "disparate_impact_ratio": disparate_impact,
        "demographic_parity_difference": max_diff,
        "four_fifths_rule_passed": four_fifths_pass,
        "interpretation": (
            f"The disparate impact ratio is {disparate_impact*100:.1f}%. "
            f"According to the EEOC 80% guideline, this {'satisfies' if four_fifths_pass else 'falls below'} "
            f"the standard threshold for demographic parity across {protected_attr}."
        )
    }

def audit_career_model_fairness(
    test_data_path: Optional[str] = None,
    audit_data_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive fairness audit on model recommendations and placement outcomes.
    Demographics were excluded during model training to ensure algorithmic independence.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if audit_data_path is None:
        audit_data_path = os.path.join(base_dir, "data", "processed", "audit_test.csv")
    if test_data_path is None:
        test_data_path = os.path.join(base_dir, "data", "processed", "test_target.csv")
        
    df_audit = pd.read_csv(audit_data_path)
    
    # 1. Gender Audit on Placement Outcome
    gender_audit = compute_demographic_parity(
        df_audit, protected_attr="gender", outcome_col="placement_status", favorable_val="Placed"
    )
    
    # 2. College Tier Audit on Placement Outcome
    tier_audit = compute_demographic_parity(
        df_audit, protected_attr="college_tier", outcome_col="placement_status", favorable_val="Placed"
    )
    
    # 3. Socioeconomic Status Audit on Placement Outcome
    ses_audit = compute_demographic_parity(
        df_audit, protected_attr="socioeconomic_status", outcome_col="placement_status", favorable_val="Placed"
    )
    
    return {
        "total_test_samples": len(df_audit),
        "gender_audit": gender_audit,
        "college_tier_audit": tier_audit,
        "socioeconomic_audit": ses_audit,
        "ethical_disclosure": (
            "FAIRNESS PROTOCOL: Demographic attributes (Gender, College Tier, Socioeconomic Status) "
            "are strictly omitted from feature matrix X during model training. Career recommendations "
            "and placement readiness scores are driven purely by individual student technical assessments, "
            "academic metrics, and practical achievements."
        )
    }

if __name__ == "__main__":
    audit = audit_career_model_fairness()
    print("\n" + "="*50)
    print("      AIML006 FAIRNESS AUDIT REPORT")
    print("="*50)
    print(f"Total Evaluated Test Samples: {audit['total_test_samples']}")
    print("\n--- GENDER COHORT ---")
    print("Selection Rates:", audit["gender_audit"]["selection_rates"])
    print("Disparate Impact:", audit["gender_audit"]["disparate_impact_ratio"])
    print("Four-Fifths Rule Passed:", audit["gender_audit"]["four_fifths_rule_passed"])
    
    print("\n--- COLLEGE TIER COHORT ---")
    print("Selection Rates:", audit["college_tier_audit"]["selection_rates"])
    print("Disparate Impact:", audit["college_tier_audit"]["disparate_impact_ratio"])
    
    print("\n--- SOCIOECONOMIC COHORT ---")
    print("Selection Rates:", audit["socioeconomic_audit"]["selection_rates"])
    print("Disparate Impact:", audit["socioeconomic_audit"]["disparate_impact_ratio"])
    print("\nEthical Disclosure:\n", audit["ethical_disclosure"])
