import os
from typing import TypedDict
from pymongo import MongoClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

# Import modular actions
from agents.search_agent import search_freelancers
from agents.Proposal.proposal_agent import generate_draft_action

load_dotenv()

# ==========================================
# 1. Database & Checkpointer Setup
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

# Persistent checkpointing in MongoDB
checkpointer = MongoDBSaver(client, db_name="ProposalAgent", collection_name="proposal_checkpoints")

# ==========================================
# 2. Define the State
# ==========================================
class ProposalState(TypedDict):
    user_id: str
    job_title: str
    job_description: str
    user_prompt: str
    resume_context: str
    current_draft: str
    is_accepted: bool

# ==========================================
# 3. Define the Nodes
# ==========================================
def fetch_rag_node(state: ProposalState):
    """Node: Fetches RAG context only if it's the first run."""
    if state.get("resume_context"):
        print("⏭️ RAG context already exists in state. Skipping fetch.")
        return {} 

    print(f"🔍 Fetching RAG context for user {state['user_id']}...")
    try:
        rag_results = search_freelancers(
            query=state['job_description'], 
            limit=3, 
            user_id=state['user_id']
        )
        resume_context = "\n".join([res["content"] for res in rag_results])
        if not resume_context:
            resume_context = "Freelancer profile details not found."
    except Exception as e:
        print(f"❌ Error fetching RAG: {e}")
        resume_context = "Error retrieving background context."
        
    return {"resume_context": resume_context}

def generate_draft_node(state: ProposalState):
    """Node: Delegates to the LLM agent to generate/revise text."""
    print("🧠 Processing Proposal Draft...")
    
    new_draft = generate_draft_action(
        job_title=state.get("job_title", ""),
        job_description=state.get("job_description", ""),
        resume_context=state.get("resume_context", ""),
        user_prompt=state.get("user_prompt", ""),
        current_draft=state.get("current_draft", "")
    )
    
    return {"current_draft": new_draft}

# ==========================================
# 4. Build and Compile the Graph
# ==========================================
workflow = StateGraph(ProposalState)

workflow.add_node("rag", fetch_rag_node)
workflow.add_node("draft", generate_draft_node)

workflow.set_entry_point("rag")
workflow.add_edge("rag", "draft")
workflow.add_edge("draft", END)

# Compile with MongoDB Checkpointer for persistent memory across restarts
proposal_agent_graph = workflow.compile(checkpointer=checkpointer)