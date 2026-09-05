"""
Skill Gap Analysis Module
Compares student competencies against industry role benchmarks to identify
strengths, weaknesses, missing proficiencies, and prioritized improvement gaps.
"""

from typing import Dict, Any, List, Optional

# Industry Benchmark Profiles for each career role (expected thresholds)
ROLE_BENCHMARKS = {
    "Data Analyst": {
        "sql_score": 80,
        "excel_score": 80,
        "powerbi_score": 75,
        "python_score": 60,
        "stats_score": 65,
        "comm_score": 65,
        "aptitude_score": 70,
        "ml_score": 45,
        "projects_count": 2,
        "internships_count": 1,
        "certifications_count": 1,
        "cgpa": 7.0
    },
    "Business Analyst": {
        "comm_score": 80,
        "excel_score": 80,
        "aptitude_score": 75,
        "powerbi_score": 70,
        "sql_score": 60,
        "stats_score": 55,
        "python_score": 45,
        "ml_score": 35,
        "projects_count": 2,
        "internships_count": 1,
        "certifications_count": 1,
        "cgpa": 7.0
    },
    "Data Scientist": {
        "python_score": 82,
        "stats_score": 82,
        "ml_score": 80,
        "sql_score": 75,
        "comm_score": 65,
        "aptitude_score": 75,
        "powerbi_score": 55,
        "excel_score": 55,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 1,
        "cgpa": 7.5
    },
    "ML Intern": {
        "ml_score": 85,
        "python_score": 82,
        "stats_score": 75,
        "aptitude_score": 75,
        "sql_score": 65,
        "comm_score": 60,
        "excel_score": 50,
        "powerbi_score": 45,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 1,
        "cgpa": 7.2
    },
    "BI Analyst": {
        "powerbi_score": 85,
        "sql_score": 80,
        "excel_score": 78,
        "comm_score": 72,
        "aptitude_score": 70,
        "stats_score": 60,
        "python_score": 50,
        "ml_score": 40,
        "projects_count": 2,
        "internships_count": 1,
        "certifications_count": 2,
        "cgpa": 7.0
    },
    "Data Science Intern": {
        "python_score": 75,
        "stats_score": 72,
        "sql_score": 70,
        "ml_score": 68,
        "aptitude_score": 70,
        "comm_score": 65,
        "excel_score": 60,
        "powerbi_score": 55,
        "projects_count": 2,
        "internships_count": 0,
        "certifications_count": 1,
        "cgpa": 7.2
    }
}

SKILL_LABELS = {
    "python_score": "Python",
    "sql_score": "SQL",
    "excel_score": "Excel",
    "stats_score": "Statistics",
    "ml_score": "Machine Learning",
    "powerbi_score": "Power BI",
    "comm_score": "Communication",
    "aptitude_score": "Aptitude"
}

def analyze_skill_gaps(student_profile: Dict[str, Any], target_role: str) -> Dict[str, Any]:
    """
    Perform a granular gap analysis comparing the student's profile against the
    specified target role benchmark.
    """
    if target_role not in ROLE_BENCHMARKS:
        # Fallback to nearest default role
        target_role = "Data Analyst"
        
    benchmark = ROLE_BENCHMARKS[target_role]
    
    comparisons = []
    strong_skills = []
    weak_skills = []
    missing_skills = []
    priority_gaps = []
    
    for key, label in SKILL_LABELS.items():
        student_val = float(student_profile.get(key, 0))
        required_val = float(benchmark.get(key, 50))
        gap = required_val - student_val
        gap_pct = round((gap / required_val) * 100, 1) if required_val > 0 else 0.0
        
        entry = {
            "key": key,
            "skill": label,
            "student_score": student_val,
            "required_score": required_val,
            "gap": max(0.0, gap),
            "surplus": max(0.0, -gap),
            "gap_percentage": gap_pct
        }
        comparisons.append(entry)
        
        if student_val >= required_val:
            strong_skills.append({
                "skill": label,
                "score": student_val,
                "benchmark": required_val,
                "surplus": round(student_val - required_val, 1)
            })
        elif student_val < 40.0:
            missing_skills.append({
                "skill": label,
                "score": student_val,
                "benchmark": required_val,
                "gap": round(required_val - student_val, 1)
            })
        else:
            weak_skills.append({
                "skill": label,
                "score": student_val,
                "benchmark": required_val,
                "gap": round(required_val - student_val, 1)
            })
            
        if gap > 0:
            priority_gaps.append({
                "skill": label,
                "student_score": student_val,
                "required_score": required_val,
                "gap": round(gap, 1)
            })
            
    # Sort priority gaps by largest gap first
    priority_gaps.sort(key=lambda x: x["gap"], reverse=True)
    
    # Practical metrics comparison
    practical_gaps = {}
    for metric, req_val in [("projects_count", benchmark.get("projects_count", 2)),
                            ("internships_count", benchmark.get("internships_count", 1)),
                            ("certifications_count", benchmark.get("certifications_count", 1))]:
        curr_val = student_profile.get(metric, 0)
        practical_gaps[metric] = {
            "current": curr_val,
            "required": req_val,
            "gap": max(0, req_val - curr_val)
        }

    return {
        "target_role": target_role,
        "comparisons": comparisons,
        "strong_skills": strong_skills,
        "weak_skills": weak_skills,
        "missing_skills": missing_skills,
        "priority_gaps": priority_gaps,
        "practical_gaps": practical_gaps
    }

if __name__ == "__main__":
    sample = {
        "python_score": 60, "sql_score": 70, "excel_score": 50,
        "stats_score": 55, "ml_score": 30, "powerbi_score": 40,
        "comm_score": 65, "aptitude_score": 68
    }
    analysis = analyze_skill_gaps(sample, "Data Analyst")
    print(f"Gap Analysis for {analysis['target_role']}:")
    print(f"Strong: {[s['skill'] for s in analysis['strong_skills']]}")
    print(f"Weak: {[s['skill'] for s in analysis['weak_skills']]}")
    print(f"Missing: {[s['skill'] for s in analysis['missing_skills']]}")
    print(f"Top Priority Gap: {analysis['priority_gaps'][0] if analysis['priority_gaps'] else 'None'}")
