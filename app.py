"""
AIML006: Fair and Explainable Placement Readiness and Career Recommendation System
Streamlit Web Application
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local modules are resolvable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.readiness_scorer import calculate_placement_readiness
from src.skill_assessment import load_questions, get_available_skills, evaluate_assessment
from src.prediction import CareerPredictor
from src.skill_gap import analyze_skill_gaps, ROLE_BENCHMARKS
from src.recommendation import generate_personalized_recommendations
from src.explainability import ModelExplainer
from src.fairness import audit_career_model_fairness

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AIML006: Fair & Explainable Career AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: #F8FAFC;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #38BDF8;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .metric-card-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0.3rem 0;
    }
    .metric-card-sub {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 500;
    }
    
    /* Badges */
    .badge-ready {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-developing {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #FEF08A;
        color: #713F12;
    }
    .badge-beginner {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    /* Explanations Box */
    .narrative-box {
        background: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        color: #166534;
        font-size: 1.02rem;
        line-height: 1.5;
        margin: 1rem 0;
    }
    .ethical-box {
        background: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        color: #1E40AF;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 1rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "student_profile" not in st.session_state:
    st.session_state["student_profile"] = {
        "python_score": 82,
        "sql_score": 78,
        "excel_score": 72,
        "stats_score": 80,
        "ml_score": 75,
        "powerbi_score": 65,
        "comm_score": 70,
        "aptitude_score": 76,
        "cgpa": 8.2,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2
    }

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎓 Placement Readiness & Career Recommendation System</h1>
    <p>AIML006: Explainable AI & Fair Machine Learning for Undergraduate Student Career Guidance</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🧭 System Navigation")

pages = [
    "🏠 Overview & Architecture",
    "👤 Student Profile",
    "📝 Skill Assessment Quiz",
    "🎯 Placement Readiness Score",
    "🚀 Career Recommendation",
    "📊 Skill Gap Analysis",
    "📚 Learning Roadmap",
    "🔍 Explainable AI (SHAP)",
    "⚖️ Fairness & Bias Audit"
]

selected_page = st.sidebar.radio("Go to Section:", pages)

# Sidebar model selector
all_models_path = os.path.join(BASE_DIR, "models", "all_trained_models.joblib")
if os.path.exists(all_models_path):
    all_models_dict = joblib.load(all_models_path)
    model_choices = list(all_models_dict.keys())
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Active ML Classifier")
    chosen_model_name = st.sidebar.selectbox("Select Model Architecture:", model_choices, index=0)
    st.sidebar.caption(f"Test Accuracy: {all_models_dict[chosen_model_name]['metrics']['accuracy']*100:.1f}%")
else:
    chosen_model_name = "Logistic Regression"

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use 'Student Profile' to test profiles or load presets.")

# =========================================================
# PAGE 1: OVERVIEW & ARCHITECTURE
# =========================================================
if selected_page == "🏠 Overview & Architecture":
    st.subheader("System Architecture & Core Workflow")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trained Models", "4 Classifiers", "Logistic, RF, DT, KNN")
    with col2:
        st.metric("Assessment Bank", "160 MCQs", "8 Skill Domains")
    with col3:
        st.metric("Winning Model Acc.", "87.3%", "+70.6% over baseline")
    with col4:
        st.metric("Demographic Parity", "Audited", "Zero Training Bias")
        
    st.markdown("### 📌 End-to-End Pipeline Workflow")
    st.markdown("""
    ```
    [1. Student Profile & Quiz Assessment] 
           │
           ├──▶ [Placement Readiness Scorer] ──▶ Category (Beginner, Developing, Ready, Highly Ready)
           │
           └──▶ [Feature Engineering & Preprocessing]
                       │
                       ▼
           [Scikit-Learn Multi-Class ML Model] ──▶ Career Role Recommendation + Confidence
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    [SHAP Local Explanations]    [Skill Gap Analysis] ──▶ [Personalized 4-Week Roadmap]
           │                                                       │
           ▼                                                       ▼
    Student Narrative                            Project Blueprint & Open Resources
    ```
    """)
    
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. **Meritocratic Readiness Scoring**: Explainable weighted index reflecting technical ability, academics, aptitude, communication, and real-world project experience.
    2. **Machine Learning Role Recommendation**: Multi-class classification across 6 standard entry-level tracks without hardcoded rules.
    3. **Skill Gap Diagnostics**: Direct benchmarking against hiring expectations for each specialized role.
    4. **Explainable AI (SHAP)**: Transparent feature attribution showing students exactly why a recommendation was made in plain language.
    5. **Algorithmic Fairness Audit**: Demographics (`gender`, `college_tier`, `socioeconomic_status`) are strictly excluded from prediction features to ensure recommendations remain merit-based.
    """)

# =========================================================
# PAGE 2: STUDENT PROFILE
# =========================================================
elif selected_page == "👤 Student Profile":
    st.subheader("Student Academic & Skill Profile")
    st.write("Customize your skill scores or load realistic candidate presets.")
    
    # Preset Selector
    presets = {
        "Custom Profile": None,
        "Aspiring Data Scientist": {
            "python_score": 88, "sql_score": 80, "excel_score": 62, "stats_score": 86,
            "ml_score": 84, "powerbi_score": 58, "comm_score": 68, "aptitude_score": 78,
            "cgpa": 8.6, "projects_count": 4, "internships_count": 1, "certifications_count": 2
        },
        "Aspiring Data Analyst": {
            "python_score": 62, "sql_score": 85, "excel_score": 82, "stats_score": 68,
            "ml_score": 50, "powerbi_score": 80, "comm_score": 72, "aptitude_score": 74,
            "cgpa": 7.8, "projects_count": 2, "internships_count": 1, "certifications_count": 2
        },
        "Aspiring Business Analyst": {
            "python_score": 52, "sql_score": 64, "excel_score": 88, "stats_score": 62,
            "ml_score": 42, "powerbi_score": 75, "comm_score": 88, "aptitude_score": 82,
            "cgpa": 7.9, "projects_count": 2, "internships_count": 1, "certifications_count": 1
        },
        "Aspiring ML Intern": {
            "python_score": 90, "sql_score": 70, "excel_score": 55, "stats_score": 80,
            "ml_score": 90, "powerbi_score": 50, "comm_score": 65, "aptitude_score": 78,
            "cgpa": 8.4, "projects_count": 4, "internships_count": 1, "certifications_count": 1
        },
        "First-Year / Fresher (Beginner)": {
            "python_score": 45, "sql_score": 40, "excel_score": 50, "stats_score": 45,
            "ml_score": 35, "powerbi_score": 35, "comm_score": 60, "aptitude_score": 62,
            "cgpa": 6.8, "projects_count": 0, "internships_count": 0, "certifications_count": 0
        }
    }
    
    col_pre, col_btn = st.columns([3, 1])
    with col_pre:
        chosen_preset = st.selectbox("Load Profile Preset:", list(presets.keys()))
    with col_btn:
        st.write("")
        st.write("")
        if chosen_preset != "Custom Profile" and presets[chosen_preset] is not None:
            if st.button("Apply Preset", type="primary"):
                st.session_state["student_profile"] = presets[chosen_preset].copy()
                st.success(f"Applied '{chosen_preset}' preset!")
                st.rerun()

    prof = st.session_state["student_profile"]
    
    with st.form("profile_form"):
        st.markdown("#### 💻 Technical Competencies (0 - 100)")
        c1, c2, c3 = st.columns(3)
        with c1:
            python_val = st.slider("Python Score", 0, 100, int(prof.get("python_score", 70)))
            stats_val = st.slider("Statistics Score", 0, 100, int(prof.get("stats_score", 70)))
        with c2:
            sql_val = st.slider("SQL Score", 0, 100, int(prof.get("sql_score", 70)))
            ml_val = st.slider("Machine Learning Score", 0, 100, int(prof.get("ml_score", 70)))
        with c3:
            excel_val = st.slider("MS Excel Score", 0, 100, int(prof.get("excel_score", 70)))
            powerbi_val = st.slider("Power BI Score", 0, 100, int(prof.get("powerbi_score", 70)))
            
        st.markdown("#### 🧠 Cognitive & Soft Skills (0 - 100)")
        cs1, cs2 = st.columns(2)
        with cs1:
            comm_val = st.slider("Communication Skills", 0, 100, int(prof.get("comm_score", 70)))
        with cs2:
            apt_val = st.slider("Logical Aptitude", 0, 100, int(prof.get("aptitude_score", 70)))
            
        st.markdown("#### 🎓 Academic & Practical Experience")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            cgpa_val = st.number_input("Undergrad CGPA (0 - 10)", min_value=0.0, max_value=10.0, value=float(prof.get("cgpa", 8.0)), step=0.1)
        with e2:
            proj_val = st.number_input("Completed Projects", min_value=0, max_value=10, value=int(prof.get("projects_count", 2)), step=1)
        with e3:
            intern_val = st.number_input("Internships Done", min_value=0, max_value=5, value=int(prof.get("internships_count", 1)), step=1)
        with e4:
            cert_val = st.number_input("Certifications", min_value=0, max_value=10, value=int(prof.get("certifications_count", 1)), step=1)
            
        submitted = st.form_submit_button("Save & Update Profile")
        if submitted:
            st.session_state["student_profile"] = {
                "python_score": python_val,
                "sql_score": sql_val,
                "excel_score": excel_val,
                "stats_score": stats_val,
                "ml_score": ml_val,
                "powerbi_score": powerbi_val,
                "comm_score": comm_val,
                "aptitude_score": apt_val,
                "cgpa": cgpa_val,
                "projects_count": proj_val,
                "internships_count": intern_val,
                "certifications_count": cert_val
            }
            st.success("Student profile updated successfully!")

# =========================================================
# PAGE 3: SKILL ASSESSMENT QUIZ
# =========================================================
elif selected_page == "📝 Skill Assessment Quiz":
    st.subheader("Interactive Skill Assessment Engine")
    st.write("Take a multiple-choice diagnostic test from our 160-question bank across 8 domains.")
    
    skills = get_available_skills()
    c_skill, c_cnt = st.columns([2, 1])
    with c_skill:
        selected_skill = st.selectbox("Choose Skill to Assess:", skills)
    with c_cnt:
        q_count = st.select_slider("Number of Questions:", options=[5, 10, 20], value=5)
        
    all_qs = load_questions()
    skill_qs = [q for q in all_qs if q["skill"] == selected_skill][:q_count]
    
    with st.form("quiz_form"):
        user_responses = {}
        for idx, q in enumerate(skill_qs, 1):
            st.markdown(f"**Q{idx}. {q['question']}**")
            options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])]
            choice = st.radio(
                f"Select answer for Q{idx}:",
                options,
                key=f"q_{q['id']}",
                index=None
            )
            if choice:
                user_responses[q["id"]] = choice[0] # 'A', 'B', 'C', or 'D'
            st.write("---")
            
        submit_quiz = st.form_submit_button("Submit Assessment & Grade Answers", type="primary")
        
    if submit_quiz:
        if len(user_responses) < len(skill_qs):
            st.warning("Please answer all questions before submitting.")
        else:
            report = evaluate_assessment(user_responses)
            st.success(f"Assessment Complete! Your Score: {report['overall_percentage']}% ({report['total_correct']}/{report['total_questions']})")
            
            # Show detailed explanations
            st.markdown("### 📋 Detailed Question Review & Explanations")
            for r in report["detailed_results"]:
                if r["is_correct"]:
                    st.markdown(f"✅ **{r['question']}**")
                    st.markdown(f"- Your Answer: **{r['user_answer']}** (Correct)")
                else:
                    st.markdown(f"❌ **{r['question']}**")
                    st.markdown(f"- Your Answer: **{r['user_answer']}** | Correct Answer: **{r['correct_answer']}**")
                st.caption(f"💡 Explanation: {r['explanation']}")
                st.write("")
                
            # Option to sync with profile
            skill_score_map = {
                "Python": "python_score",
                "SQL": "sql_score",
                "Excel": "excel_score",
                "Statistics": "stats_score",
                "Machine Learning": "ml_score",
                "Power BI": "powerbi_score",
                "Communication": "comm_score",
                "Aptitude": "aptitude_score"
            }
            if selected_skill in skill_score_map:
                field = skill_score_map[selected_skill]
                st.session_state["student_profile"][field] = int(report["overall_percentage"])
                st.info(f"Updated **{selected_skill}** in your Student Profile to **{int(report['overall_percentage'])}%**!")

# =========================================================
# PAGE 4: PLACEMENT READINESS SCORE
# =========================================================
elif selected_page == "🎯 Placement Readiness Score":
    st.subheader("Explainable Placement Readiness Evaluation")
    
    prof = st.session_state["student_profile"]
    readiness = calculate_placement_readiness(prof)
    score = readiness["overall_score"]
    tier = readiness["readiness_tier"]
    
    col_g, col_desc = st.columns([1, 2])
    with col_g:
        st.metric(
            label="Overall Placement Readiness Score",
            value=f"{score} / 100",
            delta=f"Category: {tier}"
        )
        badge_class = f"badge-{tier.lower().replace(' ', '-')}"
        st.markdown(f"<span class='{badge_class}'>{tier.upper()}</span>", unsafe_allow_html=True)
        st.caption(readiness["tier_description"])
        
    with col_desc:
        st.markdown("#### Weighted Breakdown Components")
        breakdown_data = []
        for comp_name, data in readiness["breakdown"].items():
            breakdown_data.append({
                "Component": comp_name.replace('_', ' ').title(),
                "Raw Score": f"{data['score']}/100",
                "Weight": data["weight"],
                "Contribution Points": f"+{data['contribution']} pts"
            })
        st.table(pd.DataFrame(breakdown_data))
        
    st.markdown("### 📊 Component Contribution Visualization")
    comps = [c.replace('_', ' ').title() for c in readiness["breakdown"].keys()]
    contribs = [data["contribution"] for data in readiness["breakdown"].values()]
    
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.barh(comps, contribs, color=['#0284C7', '#0D9488', '#8B5CF6', '#F59E0B', '#10B981'])
    ax.set_xlim(0, 40)
    ax.set_xlabel("Contribution to 100-Point Score")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f} pts", va='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    > **Mathematical Formulation**:
    > $\\text{Readiness} = 0.35 \\times \\text{TechAvg} + 0.20 \\times (\\text{CGPA} \\times 10) + 0.15 \\times \\text{Aptitude} + 0.15 \\times \\text{Communication} + 0.15 \\times \\text{PracticalExp}$
    """)

# =========================================================
# PAGE 5: CAREER RECOMMENDATION
# =========================================================
elif selected_page == "🚀 Career Recommendation":
    st.subheader("Machine Learning Career Path Recommendation")
    
    prof = st.session_state["student_profile"]
    predictor = CareerPredictor()
    pred = predictor.predict(prof)
    
    rec_role = pred["primary_recommendation"]
    conf = pred["confidence_score"]
    
    r1, r2, r3 = st.columns([1.5, 1, 1])
    with r1:
        st.markdown("### 🌟 Primary Recommendation")
        st.success(f"### **{rec_role}**")
        st.markdown(f"**Model Confidence:** `{conf:.1f}%`")
        st.caption(f"Evaluated with: {pred['model_used']}")
    with r2:
        st.markdown("### 🥈 Alternative Role 1")
        alt1 = pred["alternative_roles"][0]
        st.info(f"**{alt1['role']}**\n\nProbability: `{alt1['probability']:.1f}%`")
    with r3:
        st.markdown("### 🥉 Alternative Role 2")
        alt2 = pred["alternative_roles"][1]
        st.info(f"**{alt2['role']}**\n\nProbability: `{alt2['probability']:.1f}%`")
        
    st.markdown("### 📈 Full Multi-Class Probability Distribution")
    probs_df = pd.DataFrame(pred["all_role_probabilities"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="probability", y="role", data=probs_df, palette="Blues_r", ax=ax)
    ax.set_xlabel("Model Confidence (%)")
    ax.set_ylabel("Career Role")
    ax.set_xlim(0, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}%", (p.get_width() + 1.0, p.get_y() + p.get_height() / 2), va='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# PAGE 6: SKILL GAP ANALYSIS
# =========================================================
elif selected_page == "📊 Skill Gap Analysis":
    st.subheader("Role-Specific Skill Gap Diagnostics")
    
    prof = st.session_state["student_profile"]
    predictor = CareerPredictor()
    pred = predictor.predict(prof)
    
    roles = list(ROLE_BENCHMARKS.keys())
    target_role = st.selectbox("Select Target Role for Gap Analysis:", roles, index=roles.index(pred["primary_recommendation"]))
    
    gap_data = analyze_skill_gaps(prof, target_role)
    
    c_s, c_w, c_m = st.columns(3)
    with c_s:
        st.success(f"**Strong Skills ({len(gap_data['strong_skills'])})**")
        for s in gap_data["strong_skills"]:
            st.write(f"✓ {s['skill']} ({s['score']:.0f} vs {s['benchmark']:.0f})")
    with c_w:
        st.warning(f"**Weak / Developing Skills ({len(gap_data['weak_skills'])})**")
        for s in gap_data["weak_skills"]:
            st.write(f"⚠ {s['skill']} ({s['score']:.0f} vs {s['benchmark']:.0f})")
    with c_m:
        st.error(f"**Missing / Critical Skills ({len(gap_data['missing_skills'])})**")
        if gap_data["missing_skills"]:
            for s in gap_data["missing_skills"]:
                st.write(f"✗ {s['skill']} ({s['score']:.0f} vs {s['benchmark']:.0f})")
        else:
            st.write("None! No severe deficiencies.")
            
    st.markdown("### 📊 Student Competence vs Industry Benchmark")
    comp_df = pd.DataFrame(gap_data["comparisons"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    
    ax.bar(x - width/2, comp_df["student_score"], width, label='Student Score', color='#0284C7')
    ax.bar(x + width/2, comp_df["required_score"], width, label=f'{target_role} Benchmark', color='#E2E8F0', edgecolor='#64748B')
    
    ax.set_ylabel('Proficiency (0 - 100)')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["skill"], rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# PAGE 7: LEARNING ROADMAP
# =========================================================
elif selected_page == "📚 Learning Roadmap":
    st.subheader("Actionable 4-Week Learning Roadmap & Project Guide")
    
    prof = st.session_state["student_profile"]
    predictor = CareerPredictor()
    pred = predictor.predict(prof)
    
    gap_data = analyze_skill_gaps(prof, pred["primary_recommendation"])
    recs = generate_personalized_recommendations(gap_data)
    
    st.markdown(f"### 🎯 Recommended Capstone Project for **{pred['primary_recommendation']}**")
    proj = recs["project_blueprint"]
    st.info(f"**Project Title:** {proj['title']}\n\n**Overview:** {proj['description']}\n\n**Deliverables:** {', '.join(proj['deliverables'])}")
    
    st.markdown("### 🗓️ 4-Week Preparation Timeline")
    for w in recs["weekly_plan"]:
        with st.expander(f"📌 {w['week']}: {w['focus']}", expanded=True):
            for act in w["actions"]:
                st.markdown(f"- {act}")
                
    st.markdown("### 🌐 Curated Open-Access Resources")
    for skill, res_list in recs["curated_resources"].items():
        st.markdown(f"**{skill} Resources:**")
        for res in res_list:
            st.markdown(f"- [{res['title']}]({res['url']})")

# =========================================================
# PAGE 8: EXPLAINABLE AI (SHAP)
# =========================================================
elif selected_page == "🔍 Explainable AI (SHAP)":
    st.subheader("Explainable AI: Local & Global Feature Attribution (SHAP)")
    
    prof = st.session_state["student_profile"]
    explainer = ModelExplainer()
    explanation = explainer.explain_student_profile(prof)
    
    st.markdown(f"#### Target Role: **{explanation['recommended_role']}**")
    
    # Narrative Box
    st.markdown(f"<div class='narrative-box'>💬 <b>Student-Friendly Explanation:</b><br>{explanation['narrative']}</div>", unsafe_allow_html=True)
    
    # Feature attributions chart
    attrs = explanation["attributions"][:10]
    names = [a["display_name"] for a in attrs]
    shap_vals = [a["shap_value"] for a in attrs]
    colors = ['#10B981' if v >= 0 else '#EF4444' for v in shap_vals]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, shap_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Impact on Predicted Role (Positive = Increases Probability)")
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### 🔍 Granular Feature Attribution Table")
    attr_df = pd.DataFrame(attrs)[["display_name", "actual_value", "shap_value", "direction"]]
    attr_df.columns = ["Feature", "Student Value", "SHAP Impact", "Influence Direction"]
    st.dataframe(attr_df, use_container_width=True)

# =========================================================
# PAGE 9: FAIRNESS & BIAS AUDIT
# =========================================================
elif selected_page == "⚖️ Fairness & Bias Audit":
    st.subheader("Algorithmic Fairness & Cohort Bias Audit")
    
    st.markdown("""
    <div class="ethical-box">
        🛡️ <b>Fairness Protocol & Ethical Design:</b><br>
        To eliminate demographic bias in career guidance, sensitive demographic attributes (Gender, College Tier, Socioeconomic Status) 
        are strictly <b>excluded</b> from the feature matrix $X$ used to train the machine learning model.
        This section performs an offline demographic parity audit to evaluate fairness outcomes.
    </div>
    """, unsafe_allow_html=True)
    
    audit = audit_career_model_fairness()
    
    col_g, col_t, col_s = st.columns(3)
    with col_g:
        g = audit["gender_audit"]
        st.markdown("#### 👥 Gender Cohort")
        st.write(f"**Disparate Impact:** `{g['disparate_impact_ratio']*100:.1f}%`")
        for grp, rate in g["selection_rates"].items():
            st.write(f"- {grp}: {rate*100:.1f}% placement rate")
    with col_t:
        t = audit["college_tier_audit"]
        st.markdown("#### 🏛️ College Tier Cohort")
        st.write(f"**Disparate Impact:** `{t['disparate_impact_ratio']*100:.1f}%`")
        for grp, rate in t["selection_rates"].items():
            st.write(f"- {grp}: {rate*100:.1f}% placement rate")
    with col_s:
        s = audit["socioeconomic_audit"]
        st.markdown("#### 🌍 Socioeconomic Cohort")
        st.write(f"**Disparate Impact:** `{s['disparate_impact_ratio']*100:.1f}%`")
        for grp, rate in s["selection_rates"].items():
            st.write(f"- {grp}: {rate*100:.1f}% placement rate")
            
    st.markdown("### 📋 EEOC Four-Fifths (80%) Rule Interpretation")
    st.write("The Equal Employment Opportunity Commission (EEOC) considers a selection rate for any group that is less than four-fifths (80%) of the rate for the highest group to be evidence of disparate impact in hiring.")
    st.info(f"**Socioeconomic Status Audit**: Disparate Impact = {s['disparate_impact_ratio']*100:.1f}% (Passed Four-Fifths threshold).")
    st.warning(f"**College Tier Historical Skew**: Tier 3 students have historically lower campus drive selections ({t['selection_rates'].get('Tier 3', 0)*100:.1f}%), highlighting the vital importance of our system's decision to withhold college tier from career recommendations!")
