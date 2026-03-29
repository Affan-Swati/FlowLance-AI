import os
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# 1. Connect to the Client
client = MongoClient(os.getenv("MONGO_URI"))

# 2. Force the database to 'test' and the collection to 'freelancer_vectors'
db = client["test"] 
vector_col = db["freelancer_vectors"]
profiles_col = db["freelancer_profiles"] 

# 3. Initialize Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model="BAAI/bge-large-en-v1.5"
)

# 4. Initialize Vector Store
vector_store = MongoDBAtlasVectorSearch(
    collection=vector_col,
    embedding=embeddings,
    index_name="vector_index"
)


def process_and_store_resume(user_id: str, resume_id: str, data: dict):
    """Parses resume data into chunks and stores it with user_id and resume_id metadata."""
    
    # Store raw JSON tied to BOTH user_id and resume_id
    profiles_col.update_one(
        {"user_id": user_id, "resume_id": resume_id}, 
        {"$set": {
                "resume_data": data,
                "createdAt": datetime.utcnow() # Add this line
            }}, 
        upsert=True
    )
    
    chunks = []
    metadata = []

    # Atomize core skills
    if data.get("skills"):
        chunks.append(f"Technical Skills: {', '.join(data['skills'])}")
        metadata.append({"user_id": user_id, "resume_id": resume_id, "type": "skills"})

    # Atomize the 'data' dictionary (Experience, Projects, etc.)
    sections = data.get("data", {})
    for section_name, content in sections.items():
        if isinstance(content, dict):
            for title, desc in content.items():
                chunks.append(f"{section_name}: {title}. Details: {desc}")
                metadata.append({"user_id": user_id, "resume_id": resume_id, "type": section_name})
        elif isinstance(content, list):
            for item in content:
                chunks.append(f"{section_name}: {str(item)}")
                metadata.append({"user_id": user_id, "resume_id": resume_id, "type": section_name})

    if chunks:
        # Only delete vectors for THIS specific resume before upserting
        vector_col.delete_many({"user_id": user_id, "resume_id": resume_id})
        vector_store.add_texts(texts=chunks, metadatas=metadata)
    
    return {"status": "success", "resume_id": resume_id, "total_chunks": len(chunks)}


def remove_resume(user_id: str, resume_id: str):
    """Deletes a specific resume's raw profile and vector data."""
    
    # Remove from raw profiles collection
    profiles_col.delete_one({"user_id": user_id, "resume_id": resume_id})
    
    # Remove all associated chunks from the vector store
    result = vector_col.delete_many({"user_id": user_id, "resume_id": resume_id})
    
    return {"status": "success", "resume_id": resume_id, "deleted_chunks": result.deleted_count}