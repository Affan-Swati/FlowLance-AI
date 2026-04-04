import os
import requests
import json
import re
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
OLLAMA_ENDPOINT = f"{BASE_URL}api/generate"
MODEL_NAME = os.getenv("GIG_MODEL_NAME", "flowlance-gig") 


class MilestoneEstimate(BaseModel):
    title: str = Field(..., description="The name of the milestone")
    description: str = Field(..., description="Brief details of what will be delivered")
    startDate: str = Field(..., description="Start date in YYYY-MM-DD format")
    dueDate: str = Field(..., description="End/Due date in YYYY-MM-DD format")
    paymentAmount: float = Field(..., description="Realistic cost estimation for this phase")

class GigEstimationResponse(BaseModel):
    milestones: List[MilestoneEstimate]


def generate_milestones_action(job_description: str, start_date: str) -> dict:
    instruction = (
        "You are an expert technical project manager and estimator. "
        "Analyze the job description and break it down into a logical sequence of distinct milestones. "
        "Create exactly 4 to 6 milestones. Ensure dates are sequential and realistic."
    )

    prompt = (
        f"Project Start Date: {start_date}\n"
        f"Job Description: {job_description}"
    )

    # Use Pydantic to generate the JSON Schema automatically
    schema = GigEstimationResponse.model_json_schema()

    payload = {
        "model": MODEL_NAME,
        "system": instruction, 
        "prompt": prompt, 
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0.2, 
            "repeat_penalty": 1.15, 
            "num_predict": 900
        }
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        
        result_data = response.json().get("response", "")
        parsed_json = json.loads(result_data)

        validated_data = GigEstimationResponse(**parsed_json)
        
        return validated_data.model_dump() # Returns a clean Python dict
        
    except Exception as e:
        print(f"❌ LLM/Validation Error: {e}")
        raise RuntimeError(f"Failed to generate valid milestones: {e}")