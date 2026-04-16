import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.getenv("PROPOSAL_MODEL_NAME", "flowlance-proposal")

def clean_output(text: str) -> str:
    """Removes any hallucinated labels or unwanted AI chatter."""
    lines = text.splitlines()
    cleaned_lines = []
    skip_prefixes = [
        "system mode:",
        "precision editor.",
        "task:",
        "here's my",
        "here is my",
        "here's the",
        "here is the",
        "proposal:",
        "refined draft:",
        "revised proposal:",
        "final draft:"
    ]

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if not cleaned_lines and any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        if not cleaned_lines and stripped == "":
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(
        r"(?i)(\bplease\b\s*)?(i hope this|let me know|sincerely|best regards|thanks|i have modified|contact me|reach out|if you require any additional information|would like to discuss).*$",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove duplicated paragraphs from the end of the response.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    deduped = []
    seen = set()
    for paragraph in paragraphs:
        if paragraph not in seen:
            deduped.append(paragraph)
            seen.add(paragraph)

    if deduped and re.match(r"(?i)^(please|if|let me know|contact|reach out|i would be happy|i look forward to|thank you)", deduped[-1]):
        deduped = deduped[:-1]

    return "\n\n".join(deduped).strip()

def build_proposal_instruction(current_draft: str) -> str:
    if current_draft:
        return (
            "You are a professional proposal editor. Use the previous draft as the starting point and apply the user's request.\n"
            "Make focused edits to improve clarity, relevance, and concision, but do not rewrite the entire proposal from scratch.\n"
            "Return only the final proposal text with a total response under 220 words.\n"
            "Keep the proposal in 2 to 4 concise paragraphs, aiming for 2 to 3 sentences per paragraph.\n"
            "If the user request asks for concision, shorten the current draft by at least one sentence and remove filler, redundancy, and resume-like detail while preserving the main message.\n"
            "If the user request asks for more persuasive language, make the wording stronger and more client-focused without expanding length.\n"
            "When refining, keep the same main points but tighten wording, reduce paragraph count only if needed, and remove any language that reads like a resume summary.\n"
            "If the draft already addresses the required deliverables, preserve that structure and only refine language or length as needed.\n"
            "Do not include headings, labels, bullets, or any meta commentary.\n"
            "Focus on the most relevant experience for the role and do not summarize the entire resume.\n"
            "Avoid unrelated projects, AI systems, or background detail; include only information directly relevant to the React frontend, animations, responsive design, modals, and Figma implementation.\n"
            "Do not produce a skills inventory or a comma-separated tool list; mention at most two relevant technologies directly tied to the solution.\n"
            "If a user request is provided, reflect that request in the revised draft. Do not return the previous draft unchanged.\n"
            "Do not reuse the previous draft verbatim; change wording and structure while preserving meaning.\n"
            "Do not include a contact invitation such as 'please feel free to contact me', 'let me know', 'contact me', or 'reach out'.\n"
            "End with one concise, confident closing sentence that reaffirms fit without asking the client to respond."
        )

    return (
        "You are an elite freelance proposal writer. Write a polished, client-facing proposal in exactly 3 concise paragraphs, each 2 to 3 sentences, with a total response under 220 words.\n"
        "Begin by addressing the client's specific project goals and requirements; do not open with a generic personal pitch.\n"
        "The first paragraph should mention the required deliverables: pixel-perfect Figma implementation, interactive animations, responsive layout, modal behavior, and smooth user interactions.\n"
        "If the user request asks for shorter text, reduce the word count while keeping the proposal clear and persuasive.\n"
        "If the user request asks for additional details, add only relevant information directly tied to the job requirements.\n"
        "Do not include headings, labels, bullets, or any meta commentary.\n"
        "Focus on the most relevant experience for the role and do not summarize the entire resume.\n"
        "Avoid unrelated projects, AI systems, or background detail; include only information directly relevant to the React frontend, animations, responsive design, modals, and Figma implementation.\n"
        "Do not produce a skills inventory or a comma-separated tool list; mention at most two relevant technologies directly tied to the solution.\n"
        "If a user request is provided, make sure the proposal reflects that request.\n"
        "Do not include a contact invitation such as 'please feel free to contact me', 'let me know', 'contact me', or 'reach out'.\n"
        "End with one concise, confident closing sentence that reaffirms fit without asking the client to respond.\n"
        "Output only the final proposal text."
    )


def generate_draft_action(job_title, job_description, resume_context, user_prompt, current_draft):
    instruction = build_proposal_instruction(current_draft)

    prompt = (
        f"{instruction}\n\n"
        f"JOB TITLE: {job_title}\n"
        f"JOB DESCRIPTION: {job_description}\n"
        f"FREELANCER PROFILE: {resume_context if resume_context else 'No profile information available.'}\n"
        f"USER REQUEST: {user_prompt if user_prompt else 'None'}\n"
    )

    if current_draft:
        prompt += "\nREFINE THE PREVIOUS DRAFT BASED ON THE USER REQUEST AND RETURN ONLY THE FINAL REVISED PROPOSAL TEXT.\n"
        prompt += f"PREVIOUS DRAFT:\n{current_draft}\n"

    prompt += "\n\nFINAL RESPONSE:"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_ctx": 8192,
            "num_predict": 260,
            "stop": ["###", "User:", "Assistant:"]
        }
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        return clean_output(raw_text)
    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        raise RuntimeError(f"Ollama Connection Error: Make sure Ollama is running at {OLLAMA_ENDPOINT}. Details: {e}")