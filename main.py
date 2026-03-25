import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uvicorn

# Import our agents and the compiled LangGraph
from agents.scanner_agent import scan_resume
from agents.rag_ingestor import process_and_store_resume
from agents.search_agent import search_freelancers
from agents.Proposal.proposal_graph import proposal_agent_graph
from agents.analytics_agent import get_market_trends, generate_career_insights, classify_user_domain

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Schemas ---

class ProposalRequest(BaseModel):
    thread_id: str                # Unique session ID from Node.js
    user_id: str
    job_title: Optional[str] = None
    job_description: Optional[str] = None
    user_prompt: Optional[str] = ""
    current_draft: Optional[str] = None

class AnalyticsPayload(BaseModel):
    freelancerProfile: Dict[str, Any]
    portfolioHistory: List[Dict[str, Any]]
    domain: Optional[str] = None
    resumeData: Optional[Dict[str, Any]] = None

# --- Routes ---

@app.post("/api/agents/resume/process")
async def process_resume_api(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        logger.info(f"Processing upload for User: {user_id}")
        file_bytes = await file.read()
        
        # 1. Parse PDF
        extracted_data = scan_resume(file_bytes)
        
        # 2. Ingest into RAG and MongoDB
        ingestion_result = process_and_store_resume(user_id, extracted_data)
        
        return {
            "status": "success",
            "user_id": user_id,
            "data": extracted_data,
            "ingestion": ingestion_result
        }
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/resume/search")
async def search_talent(query: str, limit: int = 5, user_id: str = None):
    try:
        logger.info(f"Searching for: {query}")
        results = search_freelancers(query, limit, user_id)
        
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/proposal/generate")
async def generate_proposal_api(request: ProposalRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 2. 🚀 Update the input state mapping
        input_state = {
            "user_id": request.user_id,
            "job_title": request.job_title,
            "job_description": request.job_description,
            "user_prompt": request.user_prompt
        }
        
        # Only inject the draft if the user actually sent one (during a refinement)
        if request.current_draft:
            input_state["current_draft"] = request.current_draft
            
        # 3. Invoke the graph
        # Because 'current_draft' is standard TypedDict, passing it here
        # AUTOMATICALLY overrides whatever LangGraph had saved in MongoDB!
        final_state = proposal_agent_graph.invoke(input_state, config=config)
        
        return {
            "status": "success",
            "thread_id": request.thread_id,
            "proposal": final_state.get("current_draft", "")
        }
        
    except Exception as e:
        logger.error(f"Proposal Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/analyze-portfolio")
async def analyze_portfolio_api(payload: AnalyticsPayload):
    try:
        logger.info(f"Analyzing portfolio for user: {payload.freelancerProfile.get('username')}")
        
        user_data = {
            "profile": payload.freelancerProfile,
            "history": payload.portfolioHistory,
            "resume_data": payload.resumeData
        }

        # 1. DYNAMIC DOMAIN CLASSIFICATION
        domain = payload.domain
        if not domain:
            domain = classify_user_domain(user_data)
        
        # 2. Fetch Market Trends based on the dynamically determined domain
        market_trends = get_market_trends(domain)
        
        # 3. Generate Hugging Face AI Insights
        generated_text = generate_career_insights(user_data, market_trends)
        
        # 4. Return combined package back to Node.js
        return {
            "status": "success",
            "market_trends": market_trends,
            "ai_insights_text": generated_text
        }
        
    except Exception as e:
        logger.error(f"Portfolio Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)