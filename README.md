# Fair and Explainable Placement Readiness and Career Recommendation System (AIML006)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green.svg)](https://shap.readthedocs.io/)
[![Fairness Audit](https://img.shields.io/badge/Fairness-EEOC%20Audited-purple.svg)](https://www.eeoc.gov/)
[![Tests](https://img.shields.io/badge/Tests-9%20Passed-brightgreen.svg)]()

An end-to-end Machine Learning, Explainable AI (XAI), and algorithmic fairness platform designed to assess undergraduate student competencies, evaluate placement readiness, predict optimal career paths, diagnose skill gaps, and provide personalized learning roadmaps.

---

## Table of Contents
1. [Problem Statement & Objective](#problem-statement--objective)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Dataset & Data Dictionary](#dataset--data-dictionary)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Machine Learning Methodology & Evaluation](#machine-learning-methodology--evaluation)
8. [Placement Readiness Scoring](#placement-readiness-scoring)
9. [Skill Assessment Engine](#skill-assessment-engine)
10. [Skill Gap Analysis & Role Benchmarks](#skill-gap-analysis--role-benchmarks)
11. [Personalized Learning Roadmap](#personalized-learning-roadmap)
12. [Explainable AI (SHAP Integration)](#explainable-ai-shap-integration)
13. [Fairness & Bias Audit](#fairness--bias-audit)
14. [Repository Structure](#repository-structure)
15. [Installation & How to Run Locally](#installation--how-to-run-locally)
16. [Deployment Guide (Streamlit Cloud)](#deployment-guide-streamlit-cloud)
17. [Known Limitations & Future Scope](#known-limitations--future-scope)
18. [Author & Academic Disclaimer](#author--academic-disclaimer)

---

## Problem Statement & Objective

### Problem Statement
Undergraduate students frequently struggle to transition from academic study to professional data careers due to:
- Ambiguity about which data role best suits their specific skill profile.
- Lack of transparent, standardized measures of placement readiness.
- Black-box recommendations from opaque career tools that fail to explain *why* a role was recommended.
- Potential demographic biases in historical placement processes.

### Objective
To build a transparent, fair, and mathematically explainable system that:
1. Objectively scores student placement readiness (0–100) across 5 core dimensions.
2. Predicts the most suitable entry-level career role using verified Machine Learning classifiers.
3. Diagnoses skill gaps against standardized industry role benchmarks.
4. Generates a personalized 4-week learning roadmap and portfolio project blueprint.
5. Employs SHAP (Shapley Additive exPlanations) to provide local feature attributions in student-friendly terms.
6. Conducts cohort fairness audits (demographic parity, disparate impact) ensuring zero demographic feature leakage.

---

## Key Features

- **Diagnostic MCQ Skill Assessment**: 160 questions across 8 skill domains (Python, SQL, Excel, Statistics, Machine Learning, Power BI, Communication, Aptitude).
- **Explainable Placement Readiness Index**: Mathematical multi-attribute scoring categorized into *Beginner*, *Developing*, *Placement Ready*, and *Highly Ready*.
- **Multi-Class Machine Learning Career Recommender**: Predicts across 6 career tracks (*Data Analyst*, *Business Analyst*, *Data Scientist*, *ML Intern*, *BI Analyst*, *Data Science Intern*) with confidence percentages and top alternatives.
- **Role-Specific Skill Gap Analysis**: Granular comparison of student competencies against industry benchmark profiles.
- **Actionable 4-Week Roadmap**: Week-by-week upskilling milestones, portfolio capstone project suggestions, and curated open-access learning links.
- **Explainable AI (SHAP)**: Waterfall/bar feature importance visuals with plain-English narratives explaining individual decisions.
- **Algorithmic Fairness Audit**: Disparate Impact and Demographic Parity metrics evaluating gender, college tier, and socioeconomic cohorts.
- **Interactive Streamlit Web Interface**: Complete multi-tab user experience with reactive metrics and zero hardcoded placeholder data.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Student Profile & Assessment               │
│ (8 Skill Scores, CGPA, Projects, Internships, Certs)    │
└────────────────────────────┬────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌─────────────────────────┐
│  Placement Readiness  │         │ Preprocessing & Feature │
│    Scoring Engine     │         │   Engineering Pipeline  │
└───────────┬───────────┘         └────────────┬────────────┘
            ▼                                  ▼
┌───────────────────────┐         ┌─────────────────────────┐
│ Readiness Tier & Score│         │ Multi-Class ML Model    │
│ (0-100 & 4 Categories)│         │ (LogReg, RF, DT, KNN)   │
└───────────────────────┘         └────────────┬────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
          ┌───────────────────────┐                         ┌───────────────────────┐
          │ Career Recommendation │                         │ SHAP Explainability   │
          │ & Confidence Score    │                         │ Local Attributions    │
          └───────────┬───────────┘                         └───────────┬───────────┘
                      ▼                                                 ▼
          ┌───────────────────────┐                         ┌───────────────────────┐
          │  Skill Gap Analysis   │                         │ Student-Friendly      │
          │  vs Role Benchmarks   │                         │ Plain-English Report  │
          └───────────┬───────────┘                         └───────────────────────┘
                      ▼
          ┌───────────────────────┐
          │ Personalized 4-Week   │
          │ Learning Roadmap      │
          └───────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming and analytical workflows |
| **Data Manipulation** | Pandas, NumPy | Dataset processing, matrix operations, feature creation |
| **Machine Learning** | Scikit-Learn | Preprocessing pipeline, model training, cross-validation |
| **Explainable AI** | SHAP | Local and global Shapley feature attributions |
| **Visualization** | Matplotlib, Seaborn | Distribution plots, confusion matrix, SHAP charts |
| **Web Application** | Streamlit | Responsive, interactive user interface |
| **Persistence** | Joblib, JSON | Serializing trained models, preprocessors, and question banks |
| **Quality Assurance**| Pytest | Automated unit and integration testing suite |
| **Version Control** | Git / GitHub | Code management, versioning, deployment readiness |

---

## Dataset & Data Dictionary

The project utilizes a synthetic dataset of **1,500 student records** generated via track-aligned multivariate latent skill modeling (`data/raw/student_placement_synthetic.csv`).

### Feature Overview
| Column | Type | Range | Description | Model Input? |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | String | `STU_0001-1500` | Student unique identifier | No |
| `python_score` | Int | 25 – 99 | Assessed Python proficiency | **Yes** |
| `sql_score` | Int | 25 – 99 | Assessed SQL querying competency | **Yes** |
| `excel_score` | Int | 25 – 99 | Assessed MS Excel proficiency | **Yes** |
| `stats_score` | Int | 25 – 99 | Assessed statistical knowledge | **Yes** |
| `ml_score` | Int | 25 – 99 | Assessed machine learning understanding | **Yes** |
| `powerbi_score`| Int | 25 – 99 | Assessed Power BI dashboard skill | **Yes** |
| `comm_score` | Int | 25 – 99 | Assessed communication & presentation | **Yes** |
| `aptitude_score`| Int | 25 – 99 | Assessed quantitative / logical aptitude | **Yes** |
| `cgpa` | Float | 5.50 – 9.80 | Undergraduate academic CGPA | **Yes** |
| `projects_count`| Int | 0 – 6 | Number of completed portfolio projects | **Yes** |
| `internships_count`| Int | 0 – 3 | Industry internships completed | **Yes** |
| `certifications_count`| Int | 0 – 5 | Professional certifications earned | **Yes** |
| `target_role` | String | 6 Roles | Ground-truth career role track | **Target ($Y$)** |
| `placement_status` | String | Placed / Not Placed | Final placement outcome | Audit Only |
| `gender` | String | Male, Female, Other | Demographic cohort | **Audit Only** |
| `college_tier` | String | Tier 1, Tier 2, Tier 3 | Institutional tier classification | **Audit Only** |
| `socioeconomic_status`| String | Urban, Semi-Urban, Rural | Socioeconomic background | **Audit Only** |

---

## Exploratory Data Analysis (EDA)

The full interactive EDA is available in `notebooks/01_exploratory_data_analysis.ipynb`. Key insights:
1. **Data Completeness**: Zero missing values across all 1,500 student records.
2. **Role Distribution**: Balanced representation across all 6 roles (Data Analyst: 323, Business Analyst: 311, Data Scientist: 253, ML Intern: 220, BI Analyst: 210, Data Science Intern: 183).
3. **Skill Specialization**: Boxplot analyses reveal that Data Scientists score significantly higher in Python and ML, while Data Analysts lead in SQL and Power BI.
4. **Historical Skew**: College Tier exhibits a significant correlation with raw placement rates (Tier 1: 76.1%, Tier 3: 36.4%), directly motivating the system's design to exclude college tier from career recommendations.

---

## Machine Learning Methodology & Evaluation

### Candidate Classifiers Comparison
We evaluated four multi-class classification architectures using an 80/20 stratified split:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **87.33%** | **87.05%** | **86.42%** | **86.60%** | **87.27%** | **Selected Best** |
| **Random Forest** | 80.67% | 80.38% | 79.45% | 79.74% | 80.53% | Strong Ensemble |
| **K-Nearest Neighbors** | 78.00% | 77.41% | 76.88% | 77.00% | 77.93% | Non-Parametric |
| **Decision Tree** | 67.33% | 66.82% | 65.40% | 65.93% | 67.15% | Interpretable Tree |

### Zero Data Leakage Protocol
- The `StandardScaler` and `LabelEncoder` are fitted strictly on `X_train` (1,200 rows) and only transformed on `X_test` (300 rows).
- Sensitive demographic columns (`gender`, `college_tier`, `socioeconomic_status`) are separated into an offline audit set and never provided to feature matrix $X$.

---

## Placement Readiness Scoring

The readiness engine calculates an index from 0 to 100 using a transparent weighted formula:

$$\text{Readiness Score} = 0.35 \times \text{TechAvg} + 0.20 \times (\text{CGPA} \times 10) + 0.15 \times \text{Aptitude} + 0.15 \times \text{Communication} + 0.15 \times \text{PracticalExp}$$

### Readiness Tiers
- **< 50 (Beginner)**: Foundational stage. Needs core subject coursework and basic project building.
- **50 – 69 (Developing)**: Solid foundation. Needs capstone implementation and mock interview practice.
- **70 – 84 (Placement Ready)**: Competitive candidate meeting standard entry-level hiring criteria.
- **85 – 100 (Highly Ready)**: Outstanding profile suited for premium and product analytics roles.

---

## Skill Assessment Engine

- **Question Bank**: 160 multiple-choice questions stored in `data/questions/assessment_questions.json`.
- **Skill Domains**: 8 skills $\times$ 20 questions each.
- **Grading & Diagnostic**: Immediate scoring percentage, identification of strong skills ($\ge 75\%$), moderate skills ($50-74\%$), and weak skills ($< 50\%$), with question-by-question explanations.

---

## Skill Gap Analysis & Role Benchmarks

The system evaluates candidate profiles against industry benchmarks:

| Role | Python | SQL | Excel | Stats | ML | Power BI | Comm | Aptitude | Projects | Internships |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data Analyst** | 60 | 80 | 80 | 65 | 45 | 75 | 65 | 70 | 2 | 1 |
| **Business Analyst** | 45 | 60 | 80 | 55 | 35 | 70 | 80 | 75 | 2 | 1 |
| **Data Scientist** | 82 | 75 | 55 | 82 | 80 | 55 | 65 | 75 | 3 | 1 |
| **ML Intern** | 82 | 65 | 50 | 75 | 85 | 45 | 60 | 75 | 3 | 1 |
| **BI Analyst** | 50 | 80 | 78 | 60 | 40 | 85 | 72 | 70 | 2 | 1 |
| **Data Science Intern**| 75 | 70 | 60 | 72 | 68 | 55 | 65 | 70 | 2 | 0 |

---

## Personalized Learning Roadmap

For any detected skill gaps, the system generates:
1. **Topic Checklist**: Granular technical concepts to study (e.g. SQL Window functions, Power BI DAX).
2. **Role-Specific Capstone Project**: Complete blueprint with title, problem overview, and deliverables.
3. **Structured 4-Week Schedule**: Progressive weekly milestones from core review to mock interviews.
4. **Open-Access Resources**: Verified links to official documentation, Kaggle, Mode Analytics, and Khan Academy.

---

## Explainable AI (SHAP Integration)

- **Engine**: Computes exact Shapley values for candidate feature attributions toward the recommended role.
- **Visualization**: Horizontal attribution bar chart clearly distinguishing positive drivers (green) from negative factors (red).
- **Student-Friendly Narrative**: Automatic translation into natural language:
  > *"Your strong performance in **Python Proficiency and Machine Learning** provided the strongest positive evidence toward recommending the **Data Scientist** path. On the other hand, comparatively lower scores in **MS Excel Spreadsheets** slightly held back your alignment with business reporting roles."*

---

## Fairness & Bias Audit

### Algorithmic Fairness Strategy
- Demographic attributes (`gender`, `college_tier`, `socioeconomic_status`) are **never fed to the model** ($X$).
- Career recommendations are governed purely by student competency and merit.

### EEOC Four-Fifths Rule Audit
The Equal Employment Opportunity Commission (EEOC) considers a Disparate Impact ratio $\ge 0.80$ (80%) as non-discriminatory:
- **Gender Cohort**: Selection rates: Female (56.8%) vs Male (57.8%). Equal opportunity across genders.
- **Socioeconomic Status**: Disparate impact ratio = 86.7% (Passes the 80% rule).
- **College Tier Audit**: Highlights historical discrepancies in campus recruitment (Tier 3: 36.4% vs Tier 1: 76.1%), validating our design decision to withhold institutional labels from predictive models.

---

## Repository Structure

```
aiml006-placement-career-ai/
├── app.py                             # Main interactive Streamlit web application
├── requirements.txt                   # Production Python dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   ├── student_placement_synthetic.csv  # 1,500 student records
│   │   └── data_dictionary.md               # Detailed feature definitions
│   ├── processed/
│   │   ├── train_features.csv               # Leak-free scaled training features
│   │   ├── test_features.csv                # Leak-free scaled test features
│   │   ├── train_target.csv                 # Target training labels
│   │   ├── test_target.csv                  # Target test labels
│   │   ├── audit_test.csv                   # Demographics for fairness audit
│   │   └── preprocessor.joblib              # Serialized scaler & label encoder
│   └── questions/
│       └── assessment_questions.json        # 160 multiple-choice questions
│
├── models/
│   ├── best_career_model.joblib             # Selected best classifier artifact
│   ├── all_trained_models.joblib            # Dictionary of all 4 trained models
│   ├── model_comparison_results.json        # Detailed performance metrics
│   └── confusion_matrix.png                 # Heatmap visualization of predictions
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Complete runnable EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py                    # Synthetic dataset generation engine
│   ├── data_preprocessing.py                # Preprocessing & feature engineering
│   ├── train_model.py                       # Model training, comparison & metrics
│   ├── prediction.py                        # Model inference & probabilities
│   ├── readiness_scorer.py                  # Multi-dimensional readiness scoring
│   ├── skill_assessment.py                  # Assessment engine & quiz evaluation
│   ├── skill_gap.py                         # Benchmark matrix & gap analysis
│   ├── recommendation.py                    # Personalized 4-week learning roadmap
│   ├── explainability.py                    # SHAP attributions & narratives
│   └── fairness.py                          # Demographic parity & audit metrics
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                     # 9 comprehensive pytest test cases
│
└── assets/                                  # Graphic assets and visuals
```

---

## Installation & How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/shubhamfulzalke26/aiml006-placement-career-ai.git
cd aiml006-placement-career-ai
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Automated Tests
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment Guide (Streamlit Cloud)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: complete AIML006 placement career recommendation system"
   git remote add origin https://github.com/shubhamfulzalke26/aiml006-placement-career-ai.git
   git branch -M main
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository (`aiml006-placement-career-ai`), branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!**. Streamlit Cloud installs `requirements.txt` and deploys the live app within minutes.

---

## Known Limitations & Future Scope

### Limitations
1. **Synthetic Data**: While statistically aligned with realistic student profiles, real-world deployment requires ongoing retraining on verified university placement histories.
2. **Static Benchmarks**: Industry benchmark skill profiles reflect current entry-level job markets and will require periodic annual updates.
3. **Assessment Modality**: MCQs test conceptual knowledge; live coding tests and behavioral interviews remain essential complementary evaluation stages.

### Future Scope
1. **Resume Ingestion**: Integrating PDF/Docx resume parsers to automatically populate skill profiles using NLP.
2. **Continuous Feedback Loop**: Tracking student placement outcomes post-graduation to calibrate model weights over time.
3. **Enterprise Dashboard**: A multi-user portal for university placement cells to identify at-risk cohorts and conduct campus-wide upskilling campaigns.

---

## Author & Academic Disclaimer

- **Project**: AIML006 Capstone Project
- **Target Audience**: B.Sc. Data Science / Computer Science Undergraduates
- **Ethical Statement**: This project was developed strictly for academic guidance. Demographic data is utilized solely for offline bias auditing and is explicitly withheld from individual student predictions.
