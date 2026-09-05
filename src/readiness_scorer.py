"""
Placement Readiness Scoring Module
Calculates an explainable, multi-dimensional placement readiness score (0 - 100)
and assigns student readiness tiers.
"""

from typing import Dict, Any

def calculate_placement_readiness(student_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive placement readiness score based on a transparent,
    weighted index suitable for undergraduate campus placement evaluations.
    
    Weights Breakdown (Total = 100%):
    - Technical Competence (35%): Mean of relevant technical scores
    - Academic Performance (20%): Normalized CGPA (scale of 10 -> 100)
    - Quantitative & Logical Aptitude (15%): Aptitude test score
    - Communication & Soft Skills (15%): Communication assessment score
    - Practical Exposure (15%): Composite of projects, internships, certifications
    
    Readiness Tiers:
    - < 50: Beginner (Needs foundational coursework and basic projects)
    - 50 - 69: Developing (Good foundation; requires interview polish & capstone)
    - 70 - 84: Placement Ready (Competitive for standard campus drives)
    - >= 85: Highly Ready (Outstanding profile; suited for premium & product roles)
    """
    # 1. Technical Competence (35%)
    tech_skills = [
        student_profile.get("python_score", 0),
        student_profile.get("sql_score", 0),
        student_profile.get("excel_score", 0),
        student_profile.get("stats_score", 0),
        student_profile.get("ml_score", 0),
        student_profile.get("powerbi_score", 0)
    ]
    tech_avg = sum(tech_skills) / max(len(tech_skills), 1)
    tech_component = 0.35 * tech_avg
    
    # 2. Academic Performance (20%)
    cgpa = student_profile.get("cgpa", 0.0)
    # Convert 10-point CGPA to 100-point scale
    academic_normalized = min(max(cgpa * 10.0, 0.0), 100.0)
    academic_component = 0.20 * academic_normalized
    
    # 3. Problem Solving & Aptitude (15%)
    aptitude = student_profile.get("aptitude_score", 0)
    aptitude_component = 0.15 * aptitude
    
    # 4. Communication & Articulation (15%)
    comm = student_profile.get("comm_score", 0)
    comm_component = 0.15 * comm
    
    # 5. Practical Exposure (15%)
    # Scaled out of 100 points:
    # Projects: up to 4 projects * 10 = 40 max
    # Internships: up to 2 internships * 20 = 40 max
    # Certifications: up to 2 certs * 10 = 20 max
    projects = student_profile.get("projects_count", 0)
    internships = student_profile.get("internships_count", 0)
    certifications = student_profile.get("certifications_count", 0)
    
    practical_raw = (min(projects, 4) * 10.0) + (min(internships, 2) * 20.0) + (min(certifications, 2) * 10.0)
    practical_normalized = min(practical_raw, 100.0)
    practical_component = 0.15 * practical_normalized
    
    # Overall Score (0 to 100)
    total_score = round(
        tech_component + academic_component + aptitude_component + comm_component + practical_component,
        1
    )
    total_score = min(max(total_score, 0.0), 100.0)
    
    # Categorization
    if total_score >= 85.0:
        tier = "Highly Ready"
        tier_description = "Outstanding profile. Well-positioned for premium product and specialized analytics roles."
        tier_badge = "success"
    elif total_score >= 70.0:
        tier = "Placement Ready"
        tier_description = "Competitive profile meeting standard entry-level hiring criteria for campus placement drives."
        tier_badge = "primary"
    elif total_score >= 50.0:
        tier = "Developing"
        tier_description = "Solid foundation, but requires targeted project completion and interview preparation."
        tier_badge = "warning"
    else:
        tier = "Beginner"
        tier_description = "Foundational stage. Focus on core data concepts, hands-on coding, and building first portfolio projects."
        tier_badge = "danger"
        
    return {
        "overall_score": total_score,
        "readiness_tier": tier,
        "tier_description": tier_description,
        "tier_badge": tier_badge,
        "breakdown": {
            "technical_competence": {
                "score": round(tech_avg, 1),
                "weight": "35%",
                "contribution": round(tech_component, 1)
            },
            "academic_performance": {
                "score": round(academic_normalized, 1),
                "weight": "20%",
                "contribution": round(academic_component, 1)
            },
            "quantitative_aptitude": {
                "score": round(float(aptitude), 1),
                "weight": "15%",
                "contribution": round(aptitude_component, 1)
            },
            "communication_skills": {
                "score": round(float(comm), 1),
                "weight": "15%",
                "contribution": round(comm_component, 1)
            },
            "practical_exposure": {
                "score": round(practical_normalized, 1),
                "weight": "15%",
                "contribution": round(practical_component, 1)
            }
        }
    }

if __name__ == "__main__":
    sample_student = {
        "python_score": 85,
        "sql_score": 80,
        "excel_score": 75,
        "stats_score": 80,
        "ml_score": 78,
        "powerbi_score": 65,
        "comm_score": 75,
        "aptitude_score": 82,
        "cgpa": 8.4,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2
    }
    result = calculate_placement_readiness(sample_student)
    print("Sample Student Readiness:")
    print(f"Overall Score: {result['overall_score']}/100 -> {result['readiness_tier']}")
    print(f"Description: {result['tier_description']}")
