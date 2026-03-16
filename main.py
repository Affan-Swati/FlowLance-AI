import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from agents.scanner_agent import scan_resume
from agents.rag_ingestor import process_and_store_resume
from dotenv import load_dotenv
import uvicorn

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)