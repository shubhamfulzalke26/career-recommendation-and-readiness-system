# AIML006 Project: Student Placement & Career Dataset Dictionary

## 1. Overview
- **Dataset Name**: Student Placement Readiness & Career Recommendation Synthetic Dataset
- **Nature of Data**: Synthetic (Generated using multi-variate latent skill distributions and domain career profiles)
- **Total Records**: 1,500
- **Total Features**: 18
- **Target Attribute**: `target_role` (Multi-class Career Role, 6 classes)
- **Audit Outcome**: `placement_status` (Binary: Placed / Not Placed)

---

## 2. Feature Definitions

| Column Name | Data Type | Value Range | Description | Used in ML Model? |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | String | `STU_0001` - `STU_1500` | Unique student identifier | No (Identifier) |
| `python_score` | Integer | 0 – 100 | Assessed proficiency in Python programming & libraries | **Yes (Predictor)** |
| `sql_score` | Integer | 0 – 100 | Assessed proficiency in SQL queries, joins, and indexing | **Yes (Predictor)** |
| `excel_score` | Integer | 0 – 100 | Assessed proficiency in MS Excel (VLOOKUP, Pivot, formulas) | **Yes (Predictor)** |
| `stats_score` | Integer | 0 – 100 | Assessed knowledge of statistics and probability | **Yes (Predictor)** |
| `ml_score` | Integer | 0 – 100 | Assessed understanding of core Machine Learning algorithms | **Yes (Predictor)** |
| `powerbi_score` | Integer | 0 – 100 | Assessed competency in Power BI / Tableau data visualization | **Yes (Predictor)** |
| `comm_score` | Integer | 0 – 100 | Assessed communication, presentation, and articulation skills | **Yes (Predictor)** |
| `aptitude_score` | Integer | 0 – 100 | Assessed quantitative aptitude and logical reasoning | **Yes (Predictor)** |
| `cgpa` | Float | 5.00 – 10.00 | Cumulative Grade Point Average (Undergraduate) | **Yes (Predictor)** |
| `projects_count` | Integer | 0 – 6 | Number of completed academic / capstone / portfolio projects | **Yes (Predictor)** |
| `internships_count` | Integer | 0 – 3 | Number of industry internships completed | **Yes (Predictor)** |
| `certifications_count` | Integer | 0 – 5 | Number of verified technical / professional certifications | **Yes (Predictor)** |
| `target_role` | String | 6 Roles | Ground truth suitable career role | **Yes (Target $Y$)** |
| `placement_status` | String | Placed, Not Placed | Final placement outcome status | No (Fairness outcome) |
| `gender` | String | Male, Female, Other | Student reported gender | **No (Fairness audit only)** |
| `college_tier` | String | Tier 1, Tier 2, Tier 3 | Institutional classification of undergraduate college | **No (Fairness audit only)** |
| `socioeconomic_status` | String | Urban, Semi-Urban, Rural | Socioeconomic background classification | **No (Fairness audit only)** |

---

## 3. Career Role Profiles (Target Classes)

1. **Data Analyst**:
   - Primary competencies: High SQL, Excel, and Power BI; moderate Python and Statistics.
   - Core focus: Business reporting, dashboarding, KPI analysis, and ad-hoc data queries.

2. **Business Analyst**:
   - Primary competencies: High Communication, Excel, Aptitude, and Power BI; moderate SQL.
   - Core focus: Requirements gathering, stakeholder engagement, workflow analysis, and strategic insights.

3. **Data Scientist**:
   - Primary competencies: High Python, Statistics, Machine Learning, and SQL; strong CGPA and project depth.
   - Core focus: Statistical modeling, predictive analytics, hypothesis testing, and feature engineering.

4. **ML Intern**:
   - Primary competencies: High Machine Learning, Python, and portfolio project count; solid statistics.
   - Core focus: Model experimentation, deep learning fundamentals, model evaluation, and deployment basics.

5. **BI Analyst**:
   - Primary competencies: High Power BI / Tableau, SQL, Excel, and solid communication.
   - Core focus: Enterprise reporting pipelines, DAX calculations, semantic models, and executive dashboards.

6. **Data Science Intern**:
   - Primary competencies: Solid Python fundamentals, Statistics, SQL, and strong academic CGPA.
   - Core focus: Data cleaning, exploratory data analysis, assisting senior data scientists, and learning ML workflows.

---

## 4. Algorithmic Fairness Disclaimer
Attributes `gender`, `college_tier`, and `socioeconomic_status` are explicitly withheld from model training and feature sets ($X$). They are retained purely to conduct offline demographic parity and disparate impact audits, ensuring model recommendations are meritocratic and free from direct demographic bias.
