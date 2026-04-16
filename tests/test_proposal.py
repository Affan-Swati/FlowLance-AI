import requests
import time
import uuid

# Update this if your FastAPI server runs on a different port
API_URL = "http://localhost:8000/api/agents/proposal/generate"

# Generate a unique session ID for each test run to ensure a fresh LangGraph state
SESSION_THREAD_ID = f"test-thread-{uuid.uuid4().hex[:8]}"
USER_ID = "68fcc512edb10f4d1fc632e1"

def test_proposal_agent():
    print(f"🔄 Starting session with Thread ID: {SESSION_THREAD_ID}")
    
    # ==========================================
    # TEST 1: Initial Draft Generation
    # ==========================================
    print("\n🚀 --- TEST 1: Initial Draft Generation ---")
    payload_1 = {
        "thread_id": SESSION_THREAD_ID,
        "user_id": USER_ID,
        "job_title": "Frontend Developer (React)",
        "job_description": """We are looking for an experienced Frontend Developer to join a short-term project focused on building a highly interactive social casino platform based on a detailed Figma design.  

The project requires strong attention to detail, high-quality implementation, and a focus on dynamic UI and smooth animations. You will collaborate with another frontend developer.

Responsibilities
	•	Develop a pixel-perfect interface based on Figma designs
	•	Implement all required pages and UI components
	•	Build a fully interactive user interface (hover states, transitions, user interactions)
	•	Implement modal windows with proper behavior and animations
	•	Create smooth animations and micro-interactions to enhance the user experience
	•	Ensure responsive layout across desktop, tablet, and mobile
	•	Prepare the frontend for backend API integration

Requirements
	•	Solid experience with React
	•	Strong understanding of responsive and adaptive design
	•	Proven ability to deliver pixel-perfect UI
	•	Experience implementing animations and interactive elements
	•	Clean, maintainable, and well-structured code

Preferred Qualifications
	•	Experience with gaming or highly interactive interfaces
	•	Familiarity with animation libraries and visual effects

Project Details
	•	Budget: $200 (fixed price) Working alongside another frontend developer
	•	Timeline: 1 week""",
        "user_prompt": "" # Empty on first run
    }
    
    response_1 = requests.post(API_URL, json=payload_1)
    
    if response_1.status_code != 200:
        print("❌ Error:", response_1.text)
        return
        
    data_1 = response_1.json()
    initial_draft = data_1.get("proposal")
    
    print("\n✅ INITIAL DRAFT:\n")
    print(initial_draft)
    
    print("\n" + "="*50 + "\n")
    print("⏳ Waiting 3 seconds before sending refinement...\n")
    time.sleep(3)
    
    # ==========================================
    # TEST 2: Refinement (Testing Editor Mode)
    # ==========================================
    print("🚀 --- TEST 2: Refinement (Testing LangGraph Memory & Overrides) ---")
    payload_2 = {
        "thread_id": SESSION_THREAD_ID, # SAME THREAD ID!
        "user_id": USER_ID,
        "job_title": "Frontend Developer (React)",
        "job_description": """We are looking for an experienced Frontend Developer to join a short-term project focused on building a highly interactive social casino platform based on a detailed Figma design.  

The project requires strong attention to detail, high-quality implementation, and a focus on dynamic UI and smooth animations. You will collaborate with another frontend developer.

Responsibilities
	•	Develop a pixel-perfect interface based on Figma designs
	•	Implement all required pages and UI components
	•	Build a fully interactive user interface (hover states, transitions, user interactions)
	•	Implement modal windows with proper behavior and animations
	•	Create smooth animations and micro-interactions to enhance the user experience
	•	Ensure responsive layout across desktop, tablet, and mobile
	•	Prepare the frontend for backend API integration

Requirements
	•	Solid experience with React
	•	Strong understanding of responsive and adaptive design
	•	Proven ability to deliver pixel-perfect UI
	•	Experience implementing animations and interactive elements
	•	Clean, maintainable, and well-structured code

Preferred Qualifications
	•	Experience with gaming or highly interactive interfaces
	•	Familiarity with animation libraries and visual effects

Project Details
	•	Budget: $200 (fixed price) Working alongside another frontend developer
	•	Timeline: 1 week""",
        "user_prompt": "Reduce the word count please and add a bit of enthusiasm", 
        "current_draft": initial_draft # Passing the previous draft back as required by main.py
    }
    
    response_2 = requests.post(API_URL, json=payload_2)
    
    if response_2.status_code != 200:
        print("❌ Error:", response_2.text)
        return
        
    data_2 = response_2.json()
    refined_draft = data_2.get("proposal")
    
    print("\n✅ REFINED DRAFT:\n")
    print(refined_draft)

if __name__ == "__main__":
    test_proposal_agent()