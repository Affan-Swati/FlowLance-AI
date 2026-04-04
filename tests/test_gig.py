import requests
import json
import uuid

# Change this if your FastAPI server runs on a different port
API_URL = "http://localhost:8000/api/agents/gig/generate-milestones"

def test_generate_milestones():
    print("🚀 Sending test request to FlowLance Gig Estimator...")
    
    # We generate a random gig_id to act as the thread_id
    mock_gig_id = f"test_gig_{uuid.uuid4().hex[:8]}"
    
    # Mock data that mimics what Node.js will eventually send (no budget/deadline limits!)
    payload = {
        "gig_id": mock_gig_id,
        "job_description": (
            "MERN application for a local coffee shop. "
            "It needs a customer-facing menu, a shopping cart, Stripe integration for payments, "
            "and a basic admin dashboard for the owner to update menu items and view orders."
        ),
        "start_date": "2026-04-10"
    }
    
    print("\n📦 Request Payload:")
    print(json.dumps(payload, indent=2))
    print("\n⏳ Waiting for AI to estimate milestones, budgets, and timelines...")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("\n✅ SUCCESS! AI Response Received:")
            response_data = response.json()
            
            # Pretty print the resulting milestones
            print(json.dumps(response_data, indent=4))
            
            # Summarize the AI's estimation
            milestones = response_data.get("milestones", [])
            
            if milestones:
                total_calculated = sum(m.get("paymentAmount", 0) for m in milestones)
                # Grab the due date of the final milestone
                estimated_due_date = milestones[-1].get("dueDate", "Unknown")
                
                print("\n📊 AI Estimation Summary:")
                print(f"Total Estimated Value: ${total_calculated:,.2f}")
                print(f"Estimated Final Deadline: {estimated_due_date}")
                print("🎉 The AI Estimator works perfectly!")
            else:
                print("\n⚠️ Warning: AI returned an empty milestone array.")
                
        else:
            print(f"\n❌ HTTP Error {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Is your FastAPI server running on localhost:8000?")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_generate_milestones()