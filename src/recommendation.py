"""
Personalized Learning Recommendations & Roadmap Module
Generates actionable learning roadmaps, topic checklists, portfolio project blueprints,
and curated open-access resources based on identified student skill gaps.
"""

from typing import Dict, Any, List

# Curated topic checklists for bridging gaps
SKILL_TOPICS_MAP = {
    "Python": [
        "Core syntax, data structures (lists, dicts, sets, tuples), and list comprehensions",
        "Object-Oriented Programming (OOP) classes, inheritance, and magic methods",
        "NumPy array vectorization and broadcasting operations",
        "Pandas data manipulation: filtering, groupby, merges, and pivot tables",
        "Handling exceptions gracefully and structuring modular Python scripts"
    ],
    "SQL": [
        "Advanced SELECT queries, multi-table joins (INNER, LEFT, FULL OUTER)",
        "Aggregate functions with GROUP BY and HAVING filters",
        "Subqueries, Common Table Expressions (WITH clauses), and derived tables",
        "Window functions: ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD()",
        "Index optimization, query execution plans, and ACID transaction concepts"
    ],
    "Excel": [
        "Dynamic lookup functions: modern XLOOKUP() and resilient INDEX-MATCH",
        "Pivot Tables, slicers, calculated fields, and multi-dimensional summaries",
        "Logical & Conditional formulas: nested IF, AND, OR, IFERROR, COUNTIFS",
        "Data Validation rules and clean spreadsheet data modeling",
        "What-If Analysis tools: Goal Seek and Scenario Manager"
    ],
    "Statistics": [
        "Descriptive statistics: measures of central tendency (Mean, Median) and dispersion (IQR, Std Dev)",
        "Probability distributions: Normal, Binomial, Poisson, and Central Limit Theorem (CLT)",
        "Hypothesis testing: formulation of Null/Alternative hypotheses, p-values, alpha levels",
        "Parametric & non-parametric tests: Student's t-test, ANOVA, and Chi-Square test",
        "Linear regression diagnostics: R-squared, homoscedasticity, and multicollinearity (VIF)"
    ],
    "Machine Learning": [
        "Supervised vs Unsupervised learning workflows and the Bias-Variance tradeoff",
        "Data preprocessing: imputation, standard scaling, and leak-free Train-Test splits",
        "Core classifiers: Logistic Regression, Decision Trees, and Random Forests",
        "Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix",
        "Hyperparameter tuning with GridSearchCV and cross-validation pipelines"
    ],
    "Power BI": [
        "Power Query Editor: data ingestion, cleaning, pivoting, and M-language transformations",
        "Star Schema data modeling, relationship cardinality (1:*), and filter directions",
        "DAX fundamentals: CALCULATE(), RELATED(), FILTER(), ALL(), and basic aggregations",
        "Interactive dashboard design: Slicers, KPI cards, drillthroughs, and bookmarks",
        "Row-Level Security (RLS) and report publishing workflows"
    ],
    "Communication": [
        "Translating complex data metrics into executive business stories and ROI impact",
        "The STAR method (Situation, Task, Action, Result) for behavioral interview clarity",
        "Executive presentation design: the 10/20/30 slide rule and visual decluttering",
        "Active listening, constructive technical feedback, and stakeholder management",
        "Writing professional technical README files and analytical project reports"
    ],
    "Aptitude": [
        "Speed math: percentages, ratios, proportions, and weighted averages",
        "Time, speed, and distance calculations (train problems, average speeds)",
        "Work and time algebra (individual vs combined work completion)",
        "Permutations, combinations, and basic probability scenarios",
        "Logical reasoning: series completion, directional orientation, and coding-decoding"
    ]
}

# Curated free, open-access resources (no paid APIs needed)
CURATED_RESOURCES = {
    "Python": [
        {"title": "Official Python Tutorial", "url": "https://docs.python.org/3/tutorial/"},
        {"title": "Kaggle Python Course", "url": "https://www.kaggle.com/learn/python"},
        {"title": "Pandas Getting Started Guide", "url": "https://pandas.pydata.org/docs/getting_started/"}
    ],
    "SQL": [
        {"title": "Mode Analytics SQL Tutorial", "url": "https://mode.com/sql-tutorial/"},
        {"title": "SQLZoo Interactive Practice", "url": "https://sqlzoo.net/"},
        {"title": "PostgreSQL Official Documentation", "url": "https://www.postgresql.org/docs/"}
    ],
    "Excel": [
        {"title": "Microsoft Excel Official Training", "url": "https://support.microsoft.com/en-us/excel"},
        {"title": "ExcelJet Formula Reference", "url": "https://exceljet.net/"}
    ],
    "Statistics": [
        {"title": "Khan Academy College Statistics", "url": "https://www.khanacademy.org/math/statistics-probability"},
        {"title": "Penn State STAT 500 Online", "url": "https://online.stat.psu.edu/stat500/"}
    ],
    "Machine Learning": [
        {"title": "Scikit-Learn User Guide & Tutorials", "url": "https://scikit-learn.org/stable/user_guide.html"},
        {"title": "Google Machine Learning Crash Course", "url": "https://developers.google.com/machine-learning/crash-course"}
    ],
    "Power BI": [
        {"title": "Microsoft Learn Power BI Path", "url": "https://learn.microsoft.com/en-us/power-bi/"},
        {"title": "SQLBI DAX Guide", "url": "https://dax.guide/"}
    ],
    "Communication": [
        {"title": "Harvard Business Review Guide to Data Storytelling", "url": "https://hbr.org/topic/data-and-visuals"},
        {"title": "Toastmasters Public Speaking Tips", "url": "https://www.toastmasters.org/resources/public-speaking-tips"}
    ],
    "Aptitude": [
        {"title": "IndiaBIX Quantitative & Logical Reasoning", "url": "https://www.indiabix.com/"},
        {"title": "GeeksforGeeks Placement Aptitude Practice", "url": "https://www.geeksforgeeks.org/placements-gq/"}
    ]
}

# Role-specific portfolio project blueprints
PROJECT_BLUEPRINTS = {
    "Data Analyst": {
        "title": "E-Commerce Customer Churn & Cohort Retention Dashboard",
        "description": "Clean multi-table transaction data in SQL, compute monthly retention cohorts, and build an interactive Power BI dashboard tracking LTV, Churn Rate, and customer segments.",
        "deliverables": ["SQL schema and aggregation queries", "Power BI interactive report file (.pbix / PDF)", "GitHub repository with data dictionary and executive findings"]
    },
    "Business Analyst": {
        "title": "Retail Operations KPI & Market Basket Association Study",
        "description": "Analyze point-of-sale datasets using Excel Pivot Tables and Power BI. Map customer buying journeys, identify cross-sell opportunities, and write a formal business requirements document (BRD).",
        "deliverables": ["Excel financial & KPI model", "Executive summary slide deck (10 slides)", "Process flowchart and requirements document"]
    },
    "Data Scientist": {
        "title": "End-to-End Predictive Maintenance & Defect Classification",
        "description": "Perform rigorous EDA and statistical hypothesis testing on sensor logs. Build and tune an ensemble model (Random Forest / XGBoost), explain predictions using SHAP, and evaluate with PR-AUC.",
        "deliverables": ["Clean Jupyter notebook with hypothesis tests and SHAP plots", "Trained model pipeline serialized with Joblib", "Reproducible GitHub repository with requirements.txt"]
    },
    "ML Intern": {
        "title": "Production-Grade Supervised Classification API & Explainability",
        "description": "Develop a modular scikit-learn machine learning pipeline with automated data preprocessing, cross-validation, hyperparameter tuning, unit tests, and SHAP local feature attribution.",
        "deliverables": ["Modular Python codebase (`src/` architecture)", "Pytest test suite with >85% coverage", "GitHub Actions CI pipeline demo"]
    },
    "BI Analyst": {
        "title": "Enterprise Financial Performance & Executive BI Semantic Model",
        "description": "Design a Star Schema data warehouse model in Power Query. Author DAX time-intelligence measures (YTD, YoY Growth, Rolling 3-Month Averages), and configure Row-Level Security.",
        "deliverables": ["Complete Star Schema dimensional model", "DAX measure script bank", "Mobile and desktop formatted Power BI dashboards"]
    },
    "Data Science Intern": {
        "title": "Exploratory Data Analysis & Statistical Modeling Capstone",
        "description": "Ingest public healthcare or socioeconomic survey data. Perform comprehensive missing value treatment, outlier diagnostics, feature engineering, and baseline linear/logistic modeling.",
        "deliverables": ["Thorough EDA report with Seaborn visualizations", "Documented hypothesis testing write-up", "Modular data cleaning script"]
    }
}

def generate_personalized_recommendations(gap_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a tailored, explainable 4-week learning roadmap and project guide
    based on priority gaps identified in the student's profile.
    """
    target_role = gap_analysis.get("target_role", "Data Analyst")
    priority_gaps = gap_analysis.get("priority_gaps", [])
    
    # 1. Identify top 3 skills that need the most work
    top_gaps = priority_gaps[:3] if priority_gaps else []
    
    actionable_checklists = {}
    recommended_resources = {}
    
    for gap_item in top_gaps:
        skill = gap_item["skill"]
        if skill in SKILL_TOPICS_MAP:
            actionable_checklists[skill] = SKILL_TOPICS_MAP[skill]
        if skill in CURATED_RESOURCES:
            recommended_resources[skill] = CURATED_RESOURCES[skill]
            
    # Fallback if student has zero skill gaps (congratulate and provide mastery topics)
    if not top_gaps:
        actionable_checklists["Mastery"] = [
            "Contribute to open-source data science tools on GitHub",
            "Participate in competitive Kaggle competitions to refine advanced feature engineering",
            "Prepare for technical system design and live SQL whiteboard interviews"
        ]
        
    # 2. Portfolio Project Recommendation
    project_blueprint = PROJECT_BLUEPRINTS.get(
        target_role,
        PROJECT_BLUEPRINTS["Data Analyst"]
    )
    
    # 3. 4-Week Structured Roadmap
    week1_focus = top_gaps[0]["skill"] if len(top_gaps) > 0 else "Core Fundamentals"
    week2_focus = top_gaps[1]["skill"] if len(top_gaps) > 1 else (top_gaps[0]["skill"] if top_gaps else "Applied Practice")
    week3_focus = top_gaps[2]["skill"] if len(top_gaps) > 2 else "Portfolio Project Implementation"
    
    weekly_plan = [
        {
            "week": "Week 1",
            "focus": f"Intensive Upskilling in {week1_focus}",
            "actions": [
                f"Complete theoretical concepts and review top documentation for {week1_focus}",
                f"Solve 20+ hands-on practice problems on {week1_focus}",
                "Document lessons learned in a dedicated study notebook"
            ]
        },
        {
            "week": "Week 2",
            "focus": f"Targeted Improvement in {week2_focus}",
            "actions": [
                f"Work through structured tutorials and build mini-exercises in {week2_focus}",
                "Synthesize concepts by integrating with previous knowledge",
                "Take a timed self-assessment quiz to verify score improvement"
            ]
        },
        {
            "week": "Week 3",
            "focus": f"Portfolio Project: {project_blueprint['title']}",
            "actions": [
                "Acquire and clean real-world dataset matching the role profile",
                "Implement end-to-end analytical workflow or predictive model",
                "Ensure clean, modular code with comments and structured functions"
            ]
        },
        {
            "week": "Week 4",
            "focus": "GitHub Documentation, Interview Polish & Mock Drives",
            "actions": [
                "Write a professional GitHub README with diagrams and key findings",
                "Practice behavioral STAR interview questions articulating your project decisions",
                "Participate in mock technical interviews and refine placement pitch"
            ]
        }
    ]
    
    return {
        "target_role": target_role,
        "top_priority_gaps": top_gaps,
        "actionable_checklists": actionable_checklists,
        "curated_resources": recommended_resources,
        "project_blueprint": project_blueprint,
        "weekly_plan": weekly_plan
    }

if __name__ == "__main__":
    from src.skill_gap import analyze_skill_gaps
    sample = {"python_score": 50, "sql_score": 60, "excel_score": 65, "powerbi_score": 40}
    gaps = analyze_skill_gaps(sample, "Data Analyst")
    recs = generate_personalized_recommendations(gaps)
    print("Personalized Plan for:", recs["target_role"])
    print("Project:", recs["project_blueprint"]["title"])
    print("Weekly Schedule:", [w["focus"] for w in recs["weekly_plan"]])
