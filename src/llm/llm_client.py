from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def llmClient():

    grok_api_key = os.getenv("GROK_API_KEY")
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=grok_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm
