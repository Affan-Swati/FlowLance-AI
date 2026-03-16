import fitz
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser # Switched to JsonOutputParser

class FlexibleResumeSchema(BaseModel):
    # core fields we definitely want
    name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(None, description="Email address")
    skills: List[str] = Field(default_factory=list, description="List of technical/soft skills")
    
    # Everything else goes here as a dynamic dictionary
    # The LLM can create keys like 'experience', 'projects', 'education' 
    # and the value can be anything (list, string, or dict)
    data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="A dictionary containing all other sections found (experience, education, projects, etc.)"
    )

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

def scan_resume(file_bytes: bytes) -> dict:
    raw_text = extract_text_from_pdf(file_bytes)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=FlexibleResumeSchema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract all resume data. Put core info in the top level. Put all sections like Experience, Education, and Projects into the 'data' dictionary. Ensure the output is valid JSON.\n{format_instructions}"),
        ("user", "{resume_text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    return chain.invoke({"resume_text": raw_text})