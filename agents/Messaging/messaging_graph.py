import os
from typing import TypedDict, List, Dict, Any
from uuid import uuid4
from pymongo import MongoClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

from agents.Messaging.messaging_agent import analyze_message_action, draft_reply_action

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(client, db_name="MessagingAgent", collection_name="messaging_checkpoints")


class ConversationItem(TypedDict):
    sender: str
    text: str


class MessagingState(TypedDict):
    thread_id: str
    user_id: str
    client_name: str
    gig_context: str
    conversation_history: List[ConversationItem]
    latest_message: str
    sentiment: str
    intent: str
    generated_reply: str


def ensure_thread_id(state: MessagingState):
    return {"thread_id": state.get("thread_id") or f"messaging_{state.get('user_id', 'unknown')}_{uuid4()}"}


def analyze_node(state: MessagingState) -> dict:
    if state.get("sentiment") and state.get("intent"):
        return {}
    print("🧠 Analyzing message sentiment and intent...")
    analysis = analyze_message_action(
        latest_message=state.get("latest_message", ""),
        gig_context=state.get("gig_context", ""),
        client_name=state.get("client_name", "Client")
    )
    return {
        "sentiment": analysis.get("sentiment", "neutral"),
        "intent": analysis.get("intent", "unknown")
    }


def draft_reply_node(state: MessagingState) -> dict:
    print("✍️ Drafting the reply...")
    reply = draft_reply_action(
        latest_message=state.get("latest_message", ""),
        intent=state.get("intent", ""),
        gig_context=state.get("gig_context", ""),
        client_name=state.get("client_name", "Client"),
        conversation_history=state.get("conversation_history", [])
    )
    return {"generated_reply": reply}


def finalize_node(state: MessagingState) -> dict:
    print("✅ Finalizing reply output...")
    return {"thread_id": state.get("thread_id", ""), "generated_reply": state.get("generated_reply", "")}


workflow = StateGraph(MessagingState)
workflow.add_node("thread", ensure_thread_id)
workflow.add_node("analyze", analyze_node)
workflow.add_node("draft", draft_reply_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("thread")
workflow.add_edge("thread", "analyze")
workflow.add_edge("analyze", "draft")
workflow.add_edge("draft", "finalize")
workflow.add_edge("finalize", END)

messaging_agent_graph = workflow.compile(checkpointer=checkpointer)
