import os
import re
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.environ.get("MESSAGING_MODEL_NAME")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME")
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.7"))

if not MODEL_NAME:
    raise EnvironmentError("MESSAGING_MODEL_NAME must be set in the environment.")
if not GROQ_MODEL_NAME:
    raise EnvironmentError("GROQ_MODEL_NAME must be set in the environment.")

client_simulator_llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=GROQ_TEMPERATURE)


def clean_ai_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"(?i)^(assistant:|response:)", "", cleaned).strip()
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned


def parse_analysis_response(raw_text: str) -> dict:
    import json
    
    raw_text = raw_text.strip() if raw_text else ""
    
    # Try to parse as JSON first
    try:
        # Try to extract JSON object from the text
        import re
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            sentiment = parsed.get("sentiment", "").strip()
            intent = parsed.get("intent", "").strip()
            if sentiment and intent:
                return {
                    "sentiment": sentiment,
                    "intent": intent
                }
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback to parsing key-value format
    sentiment = ""
    intent = ""
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        lower = line.strip().lower()
        if lower.startswith("sentiment:"):
            sentiment = line.split(":", 1)[1].strip()
        elif lower.startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
    
    # If still empty, try to split by lines
    if not sentiment or not intent:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            sentiment = sentiment or lines[0]
            intent = intent or (lines[1] if len(lines) > 1 else "")
    
    return {
        "sentiment": sentiment or "neutral",
        "intent": intent or "unknown"
    }


def format_history(messages):
    if not messages:
        return ""

    history_lines = []
    for item in messages:
        sender = item.get("sender", "client")
        text = item.get("text", "")
        history_lines.append(f"{sender.capitalize()}: {text}")
    return "\n".join(history_lines)


def build_analysis_prompt(latest_message: str, gig_context: str, client_name: str) -> str:
    return (
        "You are a smart messaging assistant for a freelancer working on a client project. "
        "Analyze the latest client message and return a short JSON-like answer with two fields: sentiment and intent. "
        "Do not include any additional commentary.\n\n"
        f"GIG CONTEXT: {gig_context}\n"
        f"CLIENT NAME: {client_name}\n"
        f"LATEST MESSAGE: {latest_message}\n\n"
        "Output format:\n"
        "sentiment: <positive/neutral/negative/empathic observation>\n"
        "intent: <core client intent or request>"
    )


def build_reply_prompt(client_name: str, latest_message: str, intent: str, gig_context: str, history: list) -> str:
    conversation = format_history(history)
    return (
        "You are a professional freelancer assistant. The client has sent a new message and you must draft a thoughtful reply. "
        "The reply should be polite, solution-focused, and aligned to the client’s current needs. "
        "Do not add labels, headings, or signatures. Return only the response text.\n\n"
        f"CLIENT NAME: {client_name}\n"
        f"GIG CONTEXT: {gig_context}\n"
        f"CLIENT INTENT: {intent}\n"
        f"LATEST MESSAGE: {latest_message}\n"
        f"CONVERSATION HISTORY:\n{conversation}\n\n"
        "Draft a single client-facing reply that acknowledges the client, answers any question, and proposes the next step clearly."
    )


def simulate_client_message_action(gig_context: str, client_name: str, conversation_history: list = None) -> str:
    history = format_history(conversation_history or [])
    prompt = (
        f"You are a client named {client_name} who is working with a freelancer on a project. "
        "Write a short, natural message asking for an update about the project status or next steps. "
        "Keep the tone professional, friendly, and concise. Do not include labels or signatures.\n\n"
        f"CLIENT NAME: {client_name}\n"
        f"GIG CONTEXT: {gig_context}\n"
        f"CONVERSATION HISTORY:\n{history}\n"
    )
    raw_response = client_simulator_llm.invoke(prompt)
    return clean_ai_text(raw_response.content)


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_ctx": 4096,
            "num_predict": 320,
            "stop": ["###", "User:", "Assistant:"]
        }
    }
    response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get("response", "")


def analyze_message_action(latest_message: str, gig_context: str, client_name: str) -> dict:
    prompt = build_analysis_prompt(latest_message=latest_message, gig_context=gig_context or "No gig context provided.", client_name=client_name or "Client")
    raw_response = call_ollama(prompt)
    parsed = parse_analysis_response(raw_response)
    return {
        "sentiment": parsed["sentiment"],
        "intent": parsed["intent"]
    }


def draft_reply_action(latest_message: str, intent: str, gig_context: str, client_name: str, conversation_history: list) -> str:
    prompt = build_reply_prompt(
        client_name=client_name or "Client",
        latest_message=latest_message,
        intent=intent or "",
        gig_context=gig_context or "No gig context provided.",
        history=conversation_history or []
    )
    raw_response = call_ollama(prompt)
    return clean_ai_text(raw_response)
