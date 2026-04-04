import os
from typing import TypedDict, Dict, Any
from pymongo import MongoClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

from agents.Gig.gig_agent import generate_milestones_action

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(client, db_name="GigAgent", collection_name="gig_checkpoints")

class GigState(TypedDict):
    gig_id: str
    job_description: str
    start_date: str
    generated_json: Dict[str, Any]

def planner_node(state: GigState):
    print("🧠 Estimating Milestones, Budgets, and Timelines...")
    
    generated_json = generate_milestones_action(
        job_description=state.get('job_description', ""),
        start_date=state.get('start_date', "")
    )
    
    print("✅ Estimation Complete!")
    return {"generated_json": generated_json}

workflow = StateGraph(GigState)
workflow.add_node("planner", planner_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", END)

gig_agent_graph = workflow.compile(checkpointer=checkpointer)