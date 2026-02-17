def profiler_prompt(**kwargs: dict):

    
        # 3. The Profiler Prompt
    prompt = f"""
    Act as an expert Data Profiler. 
    USER CONTEXT: {kwargs['user_context']}
    TASK: {kwargs['task']} | TARGET: {kwargs['target']}

    DATA FACTS:
    - Shape: {kwargs['df'].shape}
    - Potential Primary Keys: {kwargs['potential_pks']}
    - Metadata: {kwargs['metadata']['dtypes']}
    - Nulls: {kwargs['metadata']['nulls']}
    
    SAMPLE DATA:
    {kwargs['metadata']['sample']}

    YOUR MISSION:
    1. **Column Logic**: Explain each column's significance. Label them as (ID, Feature, or Target).
    2. **Granularity**: Determine the 'Grain' of the data. Is it one row per Customer? Per Transaction? Per Timestamp?
    3. **Suitability**: Briefly state if this data grain matches the user's goal of {kwargs['task']}.

    Output in clean Markdown.
    """

    return prompt