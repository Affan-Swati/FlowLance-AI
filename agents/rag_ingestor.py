import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to the Client
client = MongoClient(os.getenv("MONGO_URI"))

# 2. Force the database to 'test' and the collection to 'freelancer_vectors'
# This overrides any defaults in the URI
db = client["test"] 
vector_col = db["freelancer_vectors"]
profiles_col = db["freelancer_profiles"] # Raw JSON will go here too

# 3. Initialize Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model="BAAI/bge-large-en-v1.5"
)

# 4. Initialize Vector Store with the EXPLICIT collection
vector_store = MongoDBAtlasVectorSearch(
    collection=vector_col,
    embedding=embeddings,
    index_name="vector_index"
)

import json

def process_and_store_resume(user_id: str, data: dict):
    profiles_col.update_one({"user_id": user_id}, {"$set": {"resume_data": data}}, upsert=True)
    chunks = []
    metadata = []

    # Atomize core skills
    if data.get("skills"):
        chunks.append(f"Technical Skills: {', '.join(data['skills'])}")
        metadata.append({"user_id": user_id, "type": "skills"})

    # Atomize the 'data' dictionary (Experience, Projects, etc.)
    sections = data.get("data", {})
    for section_name, content in sections.items():
        if isinstance(content, dict):
            for title, desc in content.items():
                # This creates a separate vector for EACH job/project
                chunks.append(f"{section_name}: {title}. Details: {desc}")
                metadata.append({"user_id": user_id, "type": section_name})
        elif isinstance(content, list):
            for item in content:
                chunks.append(f"{section_name}: {str(item)}")
                metadata.append({"user_id": user_id, "type": section_name})

    if chunks:
        vector_col.delete_many({"user_id": user_id})
        vector_store.add_texts(texts=chunks, metadatas=metadata)
    
    return {"status": "success", "total_chunks": len(chunks)}