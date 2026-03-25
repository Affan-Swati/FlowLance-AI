import os
import json
import logging
import requests
import re
from datetime import date
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

AI_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def extract_json_from_llm(response_text: str) -> dict:
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match: return json.loads(match.group())
        return json.loads(response_text.strip())
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}

def classify_user_domain(user_data: dict) -> str:
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    history = user_data.get("history", user_data.get("portfolioHistory", []))
    
    resume = user_data.get("resume_data") or {}
    resume_skills = resume.get("skills", [])
    
    if not hf_token or (not history and not resume_skills):
        return "Software Engineering"

    client = InferenceClient(AI_MODEL, token=hf_token)
    
    gigs_summary = []
    for gig in history:
        tasks = [m.get("title", "") for m in gig.get("milestones", [])]
        gigs_summary.append(f"{gig.get('gigTitle', '')} ({', '.join(tasks)})")
    
    compact_gigs = " | ".join(gigs_summary)
    compact_skills = ", ".join(resume_skills[:15]) # Top 15 skills
    
    messages = [
        {
            "role": "system", 
            "content": "You are a recruiter. Classify the specific technical profession of this freelancer (e.g., 'Machine Learning', 'Full Stack Development', 'Data Science'). Return ONLY a 1-3 word profession name. No punctuation."
        },
        {
            "role": "user", 
            "content": f"Classify profession based on Gigs: {compact_gigs}\nResume Skills: {compact_skills}"
        }
    ]

    try:
        response = client.chat_completion(messages=messages, max_tokens=15, temperature=0.1)
        domain = response.choices[0].message.content.strip()
        logger.info(f"🧠 AI Classified Profession: {domain}")
        return domain
    except:
        return "Software Engineering"

def generate_dynamic_market_charts(domain: str, current_count: int, job_descriptions: str, hf_token: str) -> dict:
    client = InferenceClient(AI_MODEL, token=hf_token)
    current_month = date.today().strftime("%b")
    
    messages = [
        {
            "role": "system", 
            "content": "You are a strict JSON data extractor. Only extract concrete software tools, programming languages, and frameworks. Reject all abstract concepts."
        },
        {
            "role": "user", 
            "content": f"""Return JSON only. Context: {domain}. Count: {current_count}.
        1. Top 6 specific technologies required (e.g. React, Python, AWS, Docker, PostgreSQL). CRITICAL: EXCLUDE abstract concepts (like Data Structures, Algorithms, Problem Solving) and soft skills. Key must be exactly 'skill' and 'demand_score' 0-100.
        2. 6-month historical trend (keys must be exactly 'month' and 'active_jobs').
        Format: {{ "skills_demand": [{{"skill": "Name", "demand_score": 85}}], "historical_trend": [{{"month": "Jan", "active_jobs": 12000}}] }}
        Jobs: {job_descriptions[:1500]}"""
        }
    ]

    try:
        response = client.chat_completion(messages=messages, max_tokens=500, temperature=0.1)
        data = extract_json_from_llm(response.choices[0].message.content)
        
        for item in data.get("skills_demand", []):
            if "skill_name" in item: item["skill"] = item.pop("skill_name")
        for item in data.get("historical_trend", []):
            if "job_count" in item: item["active_jobs"] = item.pop("job_count")
            
        return data
    except Exception as e:
        logger.error(f"Chart Error: {e}")
        return {"skills_demand": [], "historical_trend": []}

def get_market_trends(domain: str) -> dict:
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    
    if not app_id or not app_key:
        return get_fallback_market_data(domain)

    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={app_id}&app_key={app_key}&what={requests.utils.quote(domain)}&results_per_page=10"
    
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        count = data.get("count", 0)
        results = data.get("results", [])
        
        salaries = []
        for j in results:
            s_min = j.get("salary_min")
            s_max = j.get("salary_max")
            if s_min and s_max:
                salaries.append((s_min + s_max) / 2)
            elif s_max:
                salaries.append(s_max)
                
        # FIXED: Convert annual to hourly, and apply a "global freelance rate" modifier (0.7)
        hourly = [round((s / 2000) * 0.7, 2) for s in salaries] if salaries else [25, 40, 65]
        
        avg_rate = round(sum(hourly) / len(hourly), 2)
        descriptions = " ".join([j.get("description", "") for j in results])
        dynamic = generate_dynamic_market_charts(domain, count, descriptions, hf_token)

        return {
            "domain": domain,
            "metrics": {"active_market_listings": count, "market_health": "Active"},
            "charts": {
                "salary_distribution": [
                    {"tier": "Entry", "rate": min(hourly)},
                    {"tier": "Market Average", "rate": avg_rate},
                    {"tier": "Top 10%", "rate": max(hourly)}
                ],
                "skills_demand": dynamic.get("skills_demand", []),
                "historical_trend": dynamic.get("historical_trend", [])
            }
        }
    except:
        return get_fallback_market_data(domain)

def generate_career_insights(user_data: dict, market_data: dict) -> dict:
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    client = InferenceClient(AI_MODEL, token=hf_token)
    
    avg_rate = market_data["charts"]["salary_distribution"][1]["rate"]
    skills_list = market_data["charts"].get("skills_demand", [])
    top_skills = [s.get("skill", "Tech") for s in skills_list[:3]]
    
    history = user_data.get("history", user_data.get("portfolioHistory", []))
    
    # Compress Gig History
    compact_history = []
    for gig in history:
        compact_history.append({
            "title": gig.get("gigTitle"),
            "pay": gig.get("totalValue"),
            "tasks": [m.get("title") for m in gig.get("milestones", [])]
        })

    # Compress Resume Data
    resume = user_data.get("resume_data") or {}
    resume_skills = resume.get("skills", [])[:15] # Take top 15 skills to save tokens
    
    # Safely extract and stringify their education/experience without blowing up context window
    resume_background = str(resume.get("data", {}))[:1200] 

    # 🚀 FIXED PROMPT: Enforced strict 2nd person phrasing ("you") and mutually exclusive advice topics
    messages = [
        {"role": "system", "content": "You are an elite executive career advisor speaking directly to the freelancer. Always use 'you' and 'your'. NEVER use third-person terms like 'the user' or 'the freelancer'. Return ONLY JSON."},
        {"role": "user", "content": f"""
        Market Data: Average is ${avg_rate}/hr. Highest Demand Skills: {', '.join(top_skills)}. 
        
        Your Freelance Portfolio: {json.dumps(compact_history)}
        Your Actual Resume Skills: {', '.join(resume_skills)}
        Your Background (Education/Experience): {resume_background}
        
        Generate an Executive Report JSON with 4 distinct keys: 'pricing', 'positioning', 'upskill', 'strategy'. 
        Each key must be an object with 'title' and 'text' (Max 3 sentences per text). 
        
        CRITICAL - Ensure each section covers UNIQUE ground without repeating the others:
        1. 'pricing': Discuss ONLY your current financial rates versus the market average of ${avg_rate}/hr.
        2. 'positioning': Discuss ONLY how you should brand yourself to clients based on your education/experience.
        3. 'upskill': Discuss ONLY specific new tech skills you should learn next based on current market demand.
        4. 'strategy': Cross-reference your Resume against your Portfolio. If your resume shows advanced skills (like AI, PyTorch, MLOps) but your current gigs are basic, advise yourself to pivot to higher-tier contracts.
        """}
    ]

    try:
        response = client.chat_completion(messages=messages, max_tokens=800, temperature=0.3)
        return extract_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Insight Error: {e}")
        return {
            "pricing": {"title": "Rate Strategy", "text": f"Your current rates are below the ${avg_rate} market average."},
            "positioning": {"title": "Market Positioning", "text": "Align your profile with your strongest resume skills."},
            "upskill": {"title": "Technical Growth", "text": f"Focus on mastering {top_skills[0] if top_skills else 'Enterprise Tech'}."},
            "strategy": {"title": "Career Move", "text": "Leverage your academic background to secure higher-tier enterprise contracts."}
        }

def get_fallback_market_data(domain: str) -> dict:
    return {
        "domain": domain,
        "metrics": {"active_market_listings": 5000, "market_health": "Stable"},
        "charts": {
            "salary_distribution": [{"tier": "Average", "rate": 55}],
            "skills_demand": [], "historical_trend": []
        }
    }