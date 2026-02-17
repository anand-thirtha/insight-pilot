import pandas as pd
import logging
from llm.llm_client import llmClient
from prompts.profiler import profiler_prompt

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt= '%Y-%m-%d %H:%M:%S'
)

def profiler_node(state: dict):
    llm = llmClient()
    # 1. Extraction of inputs from State
    file_path = state.get("file_path")
    user_context = state.get("user_context", "No context provided")
    task = state.get("task_type", "General EDA")
    target = state.get("target_col", "Not specified")
    
    # Load data
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    # 2. Programmatic Metadata (The "Truth")
    # We find potential grains by checking columns where unique count == row count
    total_rows = len(df)
    potential_pks = [col for col in df.columns if df[col].nunique() == total_rows]
    
    metadata = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "uniques": df.nunique().to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_markdown()
    }

    # 3. The Profiler Prompt
    prompt = profiler_prompt(user_context = user_context,
                             task = task,
                             target = target,
                             potential_pks = potential_pks,
                             df = df,
                             metadata= metadata,
                             )
    logging.info("Generating data profiling...")
    response = llm.invoke(prompt)

    return {
        "data_profile": response.content,
        "messages": ["Data profiling and granularity detection completed."]
    }