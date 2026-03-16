import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
vector_col = client["test"]["freelancer_vectors"]

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model="BAAI/bge-large-en-v1.5"
)

vector_store = MongoDBAtlasVectorSearch(
    collection=vector_col,
    embedding=embeddings,
    index_name="vector_index"
)

def run_comprehensive_query(user_id, query, k=5):
    print(f"\n🚀 DEBUG: Total docs for {user_id}: {vector_col.count_documents({'user_id': user_id})}")
    
    # TEST 1: No Filter
    print("\n--- TEST 1: Searching WITHOUT user_id filter ---")
    results_raw = vector_store.similarity_search(query, k=1)
    if results_raw:
        print(f"Found something! Owner ID: {results_raw[0].metadata.get('user_id')}")
    else:
        print("Still nothing. This means the Atlas Search Index itself is broken.")

    # TEST 2: With Filter
    print(f"\n--- TEST 2: Searching WITH filter for {user_id} ---")
    results = vector_store.similarity_search(
        query,
        k=k,
        pre_filter={"user_id": {"$eq": user_id}}
    )

    if not results:
        print("❌ Filtered search failed.")
    else:
        for i, doc in enumerate(results):
            print(f"[{i+1}] {doc.page_content[:100]}...")

if __name__ == "__main__":
    UID = "test_user_123"

    # This query should now pull a mix of Projects, Skills, and Experience
    # because k=5 and the data is atomized.
    run_comprehensive_query(UID, "I need someone with MERN stack, React, and Social Media app experience")