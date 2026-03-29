import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model="BAAI/bge-large-en-v1.5"
)

client = MongoClient(os.getenv("MONGO_URI"))
vector_col = client["test"]["freelancer_vectors"]

vector_store = MongoDBAtlasVectorSearch(
    collection=vector_col,
    embedding=embeddings,
    index_name="vector_index"
)

def search_freelancers(query: str, limit: int = 5, user_id: str = None, resume_id: str = None):
    """
    Search for freelancers based on semantic overlap.
    Filters conditionally by user_id and/or resume_id.
    """
    filter_conditions = []
    
    if user_id:
        filter_conditions.append({"user_id": {"$eq": user_id}})
    if resume_id:
        filter_conditions.append({"resume_id": {"$eq": resume_id}})

    search_kwargs = {}
    if filter_conditions:
        # MongoDB Atlas Vector Search pre_filter requires a specific format for multiple conditions
        if len(filter_conditions) == 1:
            search_kwargs["pre_filter"] = filter_conditions[0]
        else:
            search_kwargs["pre_filter"] = {"$and": filter_conditions}

    # Perform similarity search
    results = vector_store.similarity_search(
        query,
        k=limit,
        **search_kwargs
    )
    
    # Format results for the API response
    formatted_results = []
    for doc in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
        
    return formatted_results