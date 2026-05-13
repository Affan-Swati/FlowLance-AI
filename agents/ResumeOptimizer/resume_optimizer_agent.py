import os
import json
import re
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.getenv("RESUME_OPTIMIZER_MODEL_NAME", "flowlance-resume-optimizer")

SYSTEM_INSTRUCTION = (
    "You are a professional resume reviewer and career coach speaking directly to the resume owner. "
    "Always use second-person pronouns: 'you', 'your'. "
    "Never say 'the candidate', 'they', or 'their' — you are talking directly to the person. "
    "Give honest, specific, and actionable feedback. "
    "CRITICAL RULE: Before writing any improvement, read the resume again and confirm the thing you are about to suggest is NOT already present. "
    "Never tell someone to add, include, or mention something that is already in their resume — doing so destroys trust. "
    "Only flag something as missing if you have verified it is genuinely absent. "
    "When something truly needs improvement, explain WHY it is weak and show EXACTLY how to fix it with a concrete rewrite. "
    "For example: if a bullet point lacks metrics, say 'This bullet reads as a duty, not an achievement. "
    "Rewrite it as: Reduced API response time by 40% by migrating to async workers.' "
    "If a section is already strong, acknowledge it briefly. "
    "Complete JSON templates exactly as instructed, replacing placeholder text with real, verified feedback."
)

# ─── Document pre-processing ──────────────────────────────────────────────────

def _parse_duration(duration: str) -> tuple:
    if not duration:
        return "", ""
    for sep in (" – ", " - ", "–", "-"):
        if sep in duration:
            parts = duration.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return duration.strip(), ""


def _to_list(val) -> list:
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str) and val.strip():
        return [v.strip() for v in val.split(",") if v.strip()]
    return []


_JOB_FIELDS    = {"position", "duration", "achievements"}
_PROJ_FIELDS   = {"description", "stack", "achievements"}
# Keys handled by dedicated preprocessors — skip in the generic pass
_KNOWN_SECTIONS = {"Education", "Work Experience", "Projects", "Skills"}


def _is_job(d) -> bool:
    return isinstance(d, dict) and ("position" in d or ("achievements" in d and "stack" not in d))


def _is_project(d) -> bool:
    return isinstance(d, dict) and ("description" in d or "stack" in d)


def _flatten_projects(name: str, d: dict, out: list):
    out.append({
        "name":    name,
        "stack":   d.get("stack", ""),
        "bullets": _to_list(d.get("achievements", [])),
    })
    for sub_key, sub_val in d.items():
        if sub_key in _PROJ_FIELDS:
            continue
        if isinstance(sub_val, dict) and _is_project(sub_val):
            _flatten_projects(sub_key, sub_val, out)


def _flatten_jobs(work_dict: dict, jobs: list, projs: list):
    for key, val in work_dict.items():
        if not isinstance(val, dict):
            continue
        if key == "Projects":
            for proj_name, proj_val in val.items():
                if isinstance(proj_val, dict):
                    _flatten_projects(proj_name, proj_val, projs)
            continue
        if _is_job(val):
            start, end = _parse_duration(val.get("duration", ""))
            jobs.append({
                "company":    key,
                "position":   val.get("position", ""),
                "start_date": start,
                "end_date":   end,
                "bullets":    _to_list(val.get("achievements", [])),
            })
            for sub_key, sub_val in val.items():
                if sub_key in _JOB_FIELDS:
                    continue
                if _is_job(sub_val):
                    sub_start, sub_end = _parse_duration(sub_val.get("duration", ""))
                    jobs.append({
                        "company":    sub_key,
                        "position":   sub_val.get("position", ""),
                        "start_date": sub_start,
                        "end_date":   sub_end,
                        "bullets":    _to_list(sub_val.get("achievements", [])),
                    })
                elif _is_project(sub_val):
                    _flatten_projects(sub_key, sub_val, projs)
        elif _is_project(val):
            _flatten_projects(key, val, projs)


def _extract_generic_lines(val, indent: int = 2) -> list:
    """Recursively flatten an arbitrary section value to readable text lines."""
    pad = " " * indent
    lines = []
    if isinstance(val, str) and val.strip():
        lines.append(f"{pad}{val.strip()}")
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, str) and item.strip():
                lines.append(f"{pad}- {item.strip()}")
            elif isinstance(item, dict):
                lines.extend(_extract_generic_lines(item, indent))
    elif isinstance(val, dict):
        for k, v in val.items():
            sub = _extract_generic_lines(v, indent + 2)
            if sub:
                lines.append(f"{pad}{k}:")
                lines.extend(sub)
    return lines


def _preprocess(resume_data: dict) -> dict:
    raw = resume_data.get("data", {})
    education: list = []
    work_experience: list = []
    projects: list = []
    extra_sections: dict = {}   # section_name -> list of text lines

    edu_section = raw.get("Education", {})
    if isinstance(edu_section, dict):
        for institution, edu in edu_section.items():
            if not isinstance(edu, dict):
                continue
            start, end = _parse_duration(edu.get("duration", ""))
            education.append({
                "institution": institution,
                "degree":      edu.get("degree", ""),
                "location":    edu.get("location", ""),
                "start_date":  start,
                "end_date":    end,
                "gpa":         edu.get("cgpa", edu.get("gpa", "")),
                "highlights":  _to_list(edu.get("elective courses", edu.get("elective_courses", ""))),
            })

    work_section = raw.get("Work Experience", {})
    if isinstance(work_section, dict):
        _flatten_jobs(work_section, work_experience, projects)

    proj_section = raw.get("Projects", {})
    if isinstance(proj_section, dict):
        for pname, pval in proj_section.items():
            if isinstance(pval, dict):
                _flatten_projects(pname, pval, projects)

    # Capture every other top-level section the PDF parser extracted
    for key, val in raw.items():
        if key in _KNOWN_SECTIONS:
            continue
        lines = _extract_generic_lines(val)
        if lines:
            extra_sections[key] = lines

    return {
        "name":            resume_data.get("name", ""),
        "email":           resume_data.get("email", ""),
        "skills":          resume_data.get("skills", []),
        "education":       education,
        "work_experience": work_experience,
        "projects":        projects,
        "extra_sections":  extra_sections,
    }


def _build_raw_text(p: dict) -> str:
    parts = []

    if name := p.get("name"):
        parts.append(f"Name: {name}")
    if email := p.get("email"):
        parts.append(f"Email: {email}")

    if skills := p.get("skills"):
        parts.append(f"\nSKILLS:\n{', '.join(str(s) for s in skills)}")

    if education := p.get("education"):
        parts.append("\nEDUCATION:")
        for edu in education:
            dur = " – ".join(filter(None, [edu["start_date"], edu["end_date"]]))
            parts.append(f"  {edu['degree']}  |  {edu['institution']}, {edu['location']}  |  {dur}")
            if edu.get("gpa"):
                parts.append(f"  GPA: {edu['gpa']}")
            if edu.get("highlights"):
                parts.append(f"  Relevant Courses: {', '.join(edu['highlights'])}")
            parts.append("")

    if work_experience := p.get("work_experience"):
        parts.append("\nWORK EXPERIENCE:")
        for job in work_experience:
            dur = " – ".join(filter(None, [job["start_date"], job["end_date"]]))
            parts.append(f"  {job['position']}  |  {job['company']}  |  {dur}")
            for b in job.get("bullets", []):
                parts.append(f"    - {b}")
            parts.append("")

    if projects := p.get("projects"):
        parts.append("\nPROJECTS:")
        for proj in projects:
            parts.append(f"  {proj['name']}")
            if proj.get("stack"):
                parts.append(f"  Stack: {proj['stack']}")
            for b in proj.get("bullets", []):
                parts.append(f"    - {b}")
            parts.append("")

    for section_name, lines in (p.get("extra_sections") or {}).items():
        parts.append(f"\n{section_name.upper()}:")
        parts.extend(lines)
        parts.append("")

    return "\n".join(parts)


_EMPTY_SECTION = lambda name: {
    "name": name,
    "status": "good_or_needs_work",
    "highlights": "FILL",
    "improvements": []
}


def _build_template(preprocessed: dict) -> str:
    """
    Build a concrete JSON template pre-seeded with the right section names.
    Giving the model a template to complete is far more reliable than asking
    a small GGUF to invent the JSON structure from scratch.
    Short placeholders conserve output tokens; instructions live in the prompt.
    """
    sections = []
    if preprocessed.get("skills"):
        sections.append(_EMPTY_SECTION("Skills"))
    if preprocessed.get("work_experience"):
        sections.append(_EMPTY_SECTION("Work Experience"))
    if preprocessed.get("education"):
        sections.append(_EMPTY_SECTION("Education"))
    if preprocessed.get("projects"):
        sections.append(_EMPTY_SECTION("Projects"))
    for name in (preprocessed.get("extra_sections") or {}):
        sections.append(_EMPTY_SECTION(name))

    template = {"overall": "FILL", "sections": sections}
    return json.dumps(template, indent=2)


# ─── JSON extraction ─────────────────────────────────────────────────────────

def _find_first_balanced_json(text: str) -> str:
    start = text.find('{')
    if start == -1:
        return ''
    depth = 0
    in_string = False
    escape_next = False
    for i, c in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return ''


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raw = _find_first_balanced_json(text)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


# ─── Output normalisation ────────────────────────────────────────────────────

def _is_placeholder(text: str) -> bool:
    return not text or text.strip().upper() in ("FILL", "FILL_IN", "FILL_IN.", "N/A", "")


def _normalize_critique(result: dict) -> dict:
    sections = []
    for s in result.get("sections", []):
        if not isinstance(s, dict):
            continue
        status = s.get("status", "good")
        if status not in ("good", "needs_work"):
            status = "needs_work" if "needs" in str(status).lower() else "good"
        # Accept both new field names and old ones for backwards compat
        highlights   = s.get("highlights",   s.get("feedback", ""))
        improvements = s.get("improvements", s.get("suggestions", []))
        if not isinstance(highlights, str):
            highlights = ""
        if _is_placeholder(highlights):
            highlights = ""
        clean_improvements = [
            sg for sg in (improvements or [])
            if isinstance(sg, str) and sg.strip() and not _is_placeholder(sg)
        ]
        sections.append({
            "name":         s.get("name", ""),
            "status":       status,
            "highlights":   highlights,
            "improvements": clean_improvements,
        })
    overall = result.get("overall", "")
    if _is_placeholder(overall):
        overall = ""
    return {
        "overall":  overall,
        "sections": sections,
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def optimize_resume(resume_data: dict) -> dict:
    """
    Analyses a parsed resume and returns section-by-section critique.
    Returns {"overall": str, "sections": [{name, status, feedback, suggestions}]}.
    Raises RuntimeError if Ollama is unreachable.
    """
    preprocessed = _preprocess(resume_data)
    raw_text     = _build_raw_text(preprocessed)
    template     = _build_template(preprocessed)

    section_names = [s["name"] for s in json.loads(template).get("sections", [])]
    section_list  = ", ".join(section_names)

    prompt = (
        "Read the resume below carefully, then complete EVERY section in the JSON template.\n\n"
        "RULES:\n"
        f"- You MUST fill in ALL {len(section_names)} sections: {section_list}. Do not skip any.\n"
        "- Replace FILL with real text. Use 'you'/'your' throughout — never 'the candidate'.\n"
        "- Replace good_or_needs_work with exactly: good OR needs_work.\n"
        "- Only reference content that actually exists in the resume.\n"
        "- highlights: 1-2 sentences on what is already strong in that section.\n"
        "- improvements: list of specific, actionable items. Each item must name the exact problem, "
        "explain why it hurts the resume, and show a concrete rewrite or example. "
        "Example: 'Your bullet \"Worked on backend\" reads as a task, not an achievement. "
        "Rewrite it as: Reduced API response time by 40% by migrating to async workers.' "
        "Leave as [] only when the section genuinely has nothing to fix.\n"
        "- ANTI-HALLUCINATION: Before writing each improvement item, scan the resume text above and confirm "
        "the thing you are about to suggest is NOT already there. "
        "Do NOT suggest adding a skill, tool, metric, section, or phrase that already appears in the resume. "
        "If you cannot point to a specific line in the resume that is actually weak, do not write that improvement.\n"
        "- overall: 2-3 sentence summary of the resume's biggest strengths and the single most important thing to fix.\n\n"
        f"RESUME:\n{raw_text}\n\n"
        f"JSON TEMPLATE TO COMPLETE:\n{template}\n\n"
        "COMPLETED JSON (double-check each improvement against the resume before writing it):"
    )

    payload = {
        "model":  MODEL_NAME,
        "system": SYSTEM_INSTRUCTION,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    0.3,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_ctx":        12288,
            "num_predict":    6000,   # raised: 4-5 detailed sections need ~4000-5000 tokens
            "stop": ["###", "Note:", "Please note:"],
        },
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        raw_response = response.json().get("response", "").strip()

        logger.info("Raw model response (%d chars, first 800): %s", len(raw_response), raw_response[:800])

        parsed   = _extract_json(raw_response)
        critique = _normalize_critique(parsed)

        # Fallback: model produced text but not parseable JSON — show it as overall
        if not critique["sections"] and raw_response:
            critique["overall"] = raw_response[:3000]

        return critique

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Ollama connection error at {OLLAMA_ENDPOINT}. "
            f"Ensure Ollama is running and model '{MODEL_NAME}' is available. Details: {e}"
        )
