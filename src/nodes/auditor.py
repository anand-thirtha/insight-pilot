import pandas as pd
import numpy as np
from llm.llm_client import llmClient
from prompts.auditor import auditor_prompt


def auditor_node(state: dict):
    # 1. Setup
    llm = llmClient()
    file_path = state.get("file_path")
    task = state.get("task_type", "General EDA")
    target = state.get("target_col")
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    # 2. MATHEMATICAL CALCULATIONS (The "Hard" Evidence)
    
    # --- Univariate ---
    # Numerical: describe() + skewness
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skewness = df[numeric_cols].skew().to_dict()
    
    # Categorical: uniques + top frequencies
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_stats = {col: {"uniques": df[col].nunique(), "top": df[col].mode()[0]} for col in cat_cols}
    
    # Dates: Range detection
    date_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist() # simplistic check
    date_ranges = {}
    for col in date_cols:
        try:
            temp_date = pd.to_datetime(df[col])
            date_ranges[col] = {"min": temp_date.min().strftime('%Y-%m-%d'), "max": temp_date.max().strftime('%Y-%m-%d')}
        except: continue

    # --- Bivariate: Target vs Rest ---
    correlations = {}
    if target in df.columns:
        if target in numeric_cols:
            # Pearson for numeric target
            correlations = df[numeric_cols].corr()[target].sort_values(ascending=False).to_dict()
        else:
            # For categorical targets, we look at group-by means or counts (simplified)
            correlations = "Categorical target: Analysis focuses on class distributions per feature."

    # --- Missing & Outliers ---
    missing = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    outlier_info = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0: outlier_info[col] = int(outlier_count)


    prompt = auditor_prompt(target = target,
                            date_ranges = date_ranges,
                            skewness = skewness,
                            cat_stats = cat_stats,
                            correlations = correlations,
                            missing = missing,
                            outlier_info = outlier_info,
                            task = task
                            )

    response = llm.invoke(prompt)

    return {
        "quality_report": response.content,
        "messages": ["Auditor provided task-specific mathematical diagnostics."]
    }