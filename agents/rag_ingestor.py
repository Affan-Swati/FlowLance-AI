import os
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

db = client["test"]
vector_col = db["freelancer_vectors"]
profiles_col = db["freelancer_profiles"]

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model="BAAI/bge-large-en-v1.5"
)

vector_store = MongoDBAtlasVectorSearch(
    collection=vector_col,
    embedding=embeddings,
    index_name="vector_index"
)


def process_and_store_resume(user_id: str, resume_id: str, data: dict):
    """
    Parses resume data into chunks and stores it with user_id and resume_id metadata.
    Does NOT overwrite gridfs_file_id or originalname set by Node.js controller.
    """

    # Upsert only the AI-extracted resume_data and timestamps
    # Node.js will separately set gridfs_file_id and originalname
    profiles_col.update_one(
        {"user_id": user_id, "resume_id": resume_id},
        {"$set": {
            "resume_data": data,
            "createdAt": datetime.utcnow()
        }},
        upsert=True
    )

    chunks = []
    metadata = []

    if data.get("skills"):
        chunks.append(f"Technical Skills: {', '.join(data['skills'])}")
        metadata.append({"user_id": user_id, "resume_id": resume_id, "type": "skills"})

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
        vector_col.delete_many({"user_id": user_id, "resume_id": resume_id})
        vector_store.add_texts(texts=chunks, metadatas=metadata)

    return {"status": "success", "resume_id": resume_id, "total_chunks": len(chunks)}


def remove_resume(user_id: str, resume_id: str):
    """Deletes a specific resume's raw profile and vector data."""

    profiles_col.delete_one({"user_id": user_id, "resume_id": resume_id})

    result = vector_col.delete_many({"user_id": user_id, "resume_id": resume_id})

    return {"status": "success", "resume_id": resume_id, "deleted_chunks": result.deleted_count}