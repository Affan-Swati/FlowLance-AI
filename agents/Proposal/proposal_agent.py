import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.getenv("MODEL_NAME", "flowlance-proposal")

def clean_output(text: str) -> str:
    """Removes any hallucinated labels or unwanted AI chatter."""
    text = re.sub(r"(?i)\*\*.*?\*\*[:\-]?\s*", "", text)
    text = re.sub(r"(?i)^[a-z\s]+[:\-]\s*", "", text)
    text = re.sub(r"(?i)^(here is|this is|updated|refined|bespoke)?\s*proposal:?\s*", "", text)
    text = re.sub(r"(?i)(i hope this|let me know|sincerely|best regards|thanks|i have modified).*$", "", text, flags=re.DOTALL)
    return text.strip()

def generate_draft_action(job_title, job_description, resume_context, user_prompt, current_draft):
    # EXPANDED FEW-SHOT EXAMPLES: 
    # Showing diverse tones (conversational, direct, problem-solving) and varied openers.
    examples = """
    === BEHAVIORAL EXAMPLES ===
    
    EXAMPLE 1: THE "RESUME DUMP" (BAD - DO NOT DO THIS)
    "I am a great fit. My Technical Skills include TypeScript, Angular, React, Node.js, Python, MongoDB, AWS, Kubernetes, Docker. At Motive I was a Software Engineering Intern. I also built FlowLance using MERN and am the Head of Team Valorant. Let me know if you want to hire me!"
    [CRITIQUE: Failed. Dumped the whole JSON array. Too short. Unprofessional. Uses irrelevant gaming experience.]

    EXAMPLE 2: CONVERSATIONAL & METRIC-DRIVEN (GOOD)
    "Hi there, I noticed you are looking for an Angular & TypeScript developer to build a highly reliable SaaS dashboard.
    During my time as a Software Engineering Intern at Motive, I architected and shipped modular solutions for the Equipment Monitoring product. By leveraging TypeScript and RESTful API integration, I delivered features that served over 120,000 businesses. 
    To ensure the high reliability your project demands, I utilized Test-Driven Development (TDD) behind feature flags. 
    I would love to bring this same scalable design to your team. Let's connect to discuss your dashboard."
    [CRITIQUE: Perfect. Friendly opener, metric-driven evidence, specific technical tools mentioned.]

    EXAMPLE 3: DIRECT & IMPACT-FOCUSED (GOOD)
    "Building a centralized dashboard requires robust backend architecture and seamless frontend execution. 
    For FlowLance, an AI-powered freelancer platform, I engineered a comprehensive MERN stack solution that automated gig management and client invoicing. By integrating AI agents for smart analytics and transaction categorization, I streamlined complex workflows into an intuitive interface. 
    I am available to bring this same full-stack expertise to your project and would be happy to discuss your specific requirements in detail."
    [CRITIQUE: Perfect. No "Hi there" greeting—jumps straight into the value prop. Uses the FlowLance project instead of Motive. Highly professional.]

    EXAMPLE 4: CONSULTATIVE & TECHNICAL (GOOD)
    "Your requirement for a context-aware AI pipeline aligns perfectly with my recent work on a Multimodal RAG & Semantic Search System. 
    I recently deployed a robust solution using LLaMA2, Sentence-BERT, and FAISS to handle both text and image queries from complex PDFs. By leveraging CoT and few-shot prompting alongside Flask and LlamaIndex, I ensured highly accurate and context-aware responses. 
    I am ready to implement a similarly efficient AI architecture for your data retrieval needs."
    [CRITIQUE: Perfect. Directly addresses the client's problem. Highly technical, focusing purely on the AI stack from the resume.]
    """

    if current_draft:
        instruction = (
            "SYSTEM MODE: PRECISION EDITOR.\n"
            "TASK: Apply the USER REQUEST to the PREVIOUS DRAFT.\n"
            "RULES:\n"
            "1. THE USER IS GOD: If the USER REQUEST asks to add a skill, tool, or experience that is NOT in the FREELANCER PROFILE, you MUST add it anyway. The User Request overrides all other data.\n"
            "2. Modify the text naturally. Do not just append a list at the end.\n"
            "3. Do not add bold labels or markdown headers.\n"
            "4. Output ONLY the finalized text. No commentary like 'I have updated...'"
        )
    else:
        instruction = (
            "SYSTEM MODE: PROPOSAL ARCHITECT.\n"
            "TASK: Write a highly professional, comprehensive 3-4 paragraph job proposal.\n"
            "COGNITIVE WORKFLOW:\n"
            "Step 1: Hook - Start by addressing the client's specific need based on the JOB DESCRIPTION. VARY YOUR OPENINGS (e.g., use a professional greeting, or jump straight into a direct value statement). Do not always say 'Hi there'.\n"
            "Step 2: Evidence - Select the single most relevant 'Work Experience' from the JSON and describe it professionally. Include metrics.\n"
            "Step 3: User Overrides - Check the USER REQUEST. If it asks to include specific skills or details not in the JSON, you MUST include them and weave them in naturally.\n"
            "Step 4: Closing - End with a confident, forward-looking call to action.\n"
            f"{examples}"
        )

    context_payload = (
        f"JOB TITLE: {job_title}\n"
        f"JOB DESCRIPTION: {job_description}\n"
        f"FREELANCER PROFILE: {resume_context}\n"
        f"USER REQUEST: {user_prompt if user_prompt else ''}\n"
        f"PREVIOUS DRAFT: {current_draft if current_draft else ''}"
    )

    payload = {
        "model": MODEL_NAME,
        "system": instruction, 
        "prompt": context_payload, 
        "stream": False,
        "options": {
            "temperature": 0.2,    
            "top_p": 0.9,
            "repeat_penalty": 1.3, 
            "stop": ["###", "User:", "Assistant:", "=== BEHAVIORAL EXAMPLES ==="]
        }
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        return clean_output(raw_text)
    except Exception as e:
        print(f"❌ LLM Generation Error: {e}")
        raise RuntimeError(f"Ollama Connection Error: Make sure Ollama is running at {OLLAMA_ENDPOINT}. Details: {e}")