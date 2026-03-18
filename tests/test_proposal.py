import requests
import time

# Update this if your FastAPI server runs on a different port
API_URL = "http://localhost:8000/api/agents/proposal/generate"

# We simulate Node.js creating a unique session ID
SESSION_THREAD_ID = "test-thread-1000"
USER_ID = "test_user_123" # Make sure this user exists in your vector DB if you want real RAG data

def test_proposal_agent():
    print("🚀 --- TEST 1: Initial Draft Generation ---")
    payload_1 = {
        "thread_id": SESSION_THREAD_ID,
        "user_id": USER_ID,
        "job_title": "Senior React Developer",
        "job_description": "We need a skilled developer to build a modern SaaS dashboard using React, TypeScript, and Tailwind. Must have experience integrating REST APIs.",
        "user_prompt": "", # Empty on first run
        "is_accepted": False
    }
    
    response_1 = requests.post(API_URL, json=payload_1)
    
    if response_1.status_code != 200:
        print("❌ Error:", response_1.text)
        return
        
    data_1 = response_1.json()
    print("\n✅ INITIAL DRAFT:\n")
    print(data_1.get("proposal"))
    
    print("\n" + "="*50 + "\n")
    print("⏳ Waiting 3 seconds before sending refinement...\n")
    time.sleep(3)
    
    print("🚀 --- TEST 2: Refinement (Testing LangGraph Memory) ---")
    payload_2 = {
        "thread_id": SESSION_THREAD_ID, # SAME THREAD ID!
        "user_id": USER_ID,
        "job_title": "Senior React Developer",
        "job_description": "We need a skilled developer to build a modern SaaS dashboard using React, TypeScript, and Tailwind. Must have experience integrating REST APIs.",
        "user_prompt": "Make this proposal much shorter and sound more aggressive and confident.", # New instruction!
        "is_accepted": False
    }
    
    response_2 = requests.post(API_URL, json=payload_2)
    data_2 = response_2.json()
    
    print("\n✅ REFINED DRAFT:\n")
    print(data_2.get("proposal"))

if __name__ == "__main__":
    test_proposal_agent()