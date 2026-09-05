"""
Skill Assessment Engine Module
Manages assessment question bank, administers tests, grades responses,
and categorizes candidate skills into strong, moderate, and weak proficiencies.
"""

import os
import json
from typing import List, Dict, Any, Optional

def load_questions(questions_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load question bank from JSON file."""
    if questions_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        questions_path = os.path.join(base_dir, "data", "questions", "assessment_questions.json")
        
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions bank not found at {questions_path}")
        
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_available_skills(questions_path: Optional[str] = None) -> List[str]:
    """Retrieve list of distinct skills available in the question bank."""
    questions = load_questions(questions_path)
    skills = sorted(list({q["skill"] for q in questions}))
    return skills

def get_questions_for_skill(skill: str, count: Optional[int] = None, questions_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch questions for a specific skill."""
    questions = load_questions(questions_path)
    matched = [q for q in questions if q["skill"].lower() == skill.lower()]
    if count is not None and count > 0:
        return matched[:count]
    return matched

def evaluate_assessment(user_answers: Dict[str, str], questions_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate candidate submissions against the answer key.
    
    Parameters:
        user_answers: Mapping of question_id -> selected_option (e.g. {"PY_01": "A"})
        
    Returns:
        Evaluation report with skill scores, overall score, question breakdown,
        and skill classification (strong, moderate, weak).
    """
    questions = load_questions(questions_path)
    q_map = {q["id"]: q for q in questions}
    
    skill_stats = {}
    detailed_results = []
    
    for q_id, user_choice in user_answers.items():
        if q_id not in q_map:
            continue
            
        q = q_map[q_id]
        skill = q["skill"]
        correct_choice = q["correct_answer"]
        is_correct = str(user_choice).strip().upper() == str(correct_choice).strip().upper()
        
        if skill not in skill_stats:
            skill_stats[skill] = {"correct": 0, "total": 0}
            
        skill_stats[skill]["total"] += 1
        if is_correct:
            skill_stats[skill]["correct"] += 1
            
        detailed_results.append({
            "id": q_id,
            "skill": skill,
            "question": q["question"],
            "options": q["options"],
            "user_answer": user_choice,
            "correct_answer": correct_choice,
            "is_correct": is_correct,
            "explanation": q["explanation"]
        })
        
    # Calculate percentages per skill
    skill_scores = {}
    for skill, data in skill_stats.items():
        pct = round((data["correct"] / data["total"]) * 100, 1) if data["total"] > 0 else 0.0
        skill_scores[skill] = {
            "score_pct": pct,
            "correct": data["correct"],
            "total": data["total"]
        }
        
    # Categorize skills
    strong_skills = []
    moderate_skills = []
    weak_skills = []
    
    for skill, data in skill_scores.items():
        score = data["score_pct"]
        if score >= 75.0:
            strong_skills.append((skill, score))
        elif score >= 50.0:
            moderate_skills.append((skill, score))
        else:
            weak_skills.append((skill, score))
            
    total_correct = sum(d["correct"] for d in skill_stats.values())
    total_questions = sum(d["total"] for d in skill_stats.values())
    overall_pct = round((total_correct / max(total_questions, 1)) * 100, 1)
    
    return {
        "overall_percentage": overall_pct,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "skill_scores": skill_scores,
        "strong_skills": sorted(strong_skills, key=lambda x: x[1], reverse=True),
        "moderate_skills": sorted(moderate_skills, key=lambda x: x[1], reverse=True),
        "weak_skills": sorted(weak_skills, key=lambda x: x[1]),
        "detailed_results": detailed_results
    }

if __name__ == "__main__":
    skills = get_available_skills()
    print("Available assessment skills:", skills)
    sample_q = get_questions_for_skill("Python", count=2)
    print(f"Sample Question: {sample_q[0]['question']}")
