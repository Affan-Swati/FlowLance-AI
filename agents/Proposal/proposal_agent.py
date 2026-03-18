import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
if not BASE_URL.endswith('/'):
    BASE_URL += '/'
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.getenv("MODEL_NAME", "flowlance-proposal")

def generate_draft_action(
    job_title: str, 
    job_description: str, 
    resume_context: str, 
    user_prompt: str, 
    current_draft: str
) -> str:
    """Formats the payload and calls the Ollama model with strict instructions and penalties."""
    
    # Construct the context payload
    context_payload = f"""JOB TITLE: {job_title}
JOB DESCRIPTION: {job_description}
FREELANCER PROFILE: {resume_context}
USER REQUEST: {user_prompt}
PREVIOUS DRAFT: {current_draft}"""

    if current_draft:
        instruction = (
            "You are a strategic proposal editor. "
            "Your goal is to REFINE the existing draft based on the user's specific request: '{user_prompt}'. "
            "STRICT RULES:\n"
            "1. Do NOT re-introduce skills or projects that were previously removed.\n"
            "2. Keep the tone sophisticated and concise.\n"
            "3. If the user asks for a specific change, implement it without altering the successful parts of the draft.\n"
            "4. Output ONLY the updated proposal text."
        )
    else:
        instruction = (
            "You are a world-class freelance business developer. "
            "Write a BESPOKE, high-impact proposal (max 200 words) for the provided Job Description.\n\n"
            "STRATEGY:\n"
            "1. ANALYSIS: Identify the top 2 pain points in the Job Description.\n"
            "2. TAILORING: From the Resume Context, select ONLY the most relevant projects"
            "and 2-3 specific tools that solve those pain points.\n"
            "3. FORBIDDEN: Do not list every technology known. Do not use 'I am thrilled', 'As a seasoned expert', "
            "or 'I believe I am a fit'. Start with a direct value proposition.\n"
            "4. TONE: Professional, confident, and brief.\n\n"
            "Output ONLY the proposal text. No placeholders like [Your Name] unless essential."
        )

    payload = {
        "model": MODEL_NAME,
        "system": instruction,
        "prompt": context_payload,
        "stream": False,
        "options": {
            "temperature": 0.3,    
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_ctx": 4096,
            "stop": ["User:", "Assistant:", "Note:"]
        }
    }
    
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status() 
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"❌ Error calling Ollama: {e}")
        return f"Error: {str(e)}"