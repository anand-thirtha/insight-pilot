
def auditor_prompt(**kwargs):

    return f"""
    Act as a Senior Data Auditor. Provide a MATHEMATICAL audit of this data for a **{kwargs['task']}** task.
    Target Variable: **{kwargs['target']}**

    RAW EVIDENCE:
    - Date Ranges: {kwargs['date_ranges']}
    - Numeric Stats (Mean/Std/Skew): {kwargs['skewness']}
    - Categorical Uniques: {kwargs['cat_stats']}
    - Bivariate (Target Correlation): {kwargs['correlations']}
    - Quality (Missing/Outliers): Missing: {kwargs['missing']}, Outliers: {kwargs['outlier_info']}

    YOUR MISSION:
    1. **Univariate Health**: Explain date coverage and flagging extreme skewness.
    2. **Bivariate Insight**: Which variables are strongest predictors or look like 'Data Leakage'?
    3. **Quality Alert**: State the impact of missing values/outliers on a {kwargs['task']} model.

    Output 4-5 bullet points. Be crisp, professional, and mathematically explanatory.
    """

    