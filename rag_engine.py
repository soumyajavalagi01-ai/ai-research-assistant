# rag_engine.py
import chromadb
import uuid
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_chroma_client():
    """Initialize ChromaDB in-memory client."""
    client = chromadb.Client()
    return client

def get_or_create_collection(client, collection_name: str = "research_data"):
    """Get or create a ChromaDB collection using default embeddings."""
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    # Use default embedding function — no sentence_transformers needed
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def store_search_results(collection, search_results: list, topic: str):
    """RAG INDEXING — Store search results as chunks in ChromaDB."""
    documents = []
    metadatas = []
    ids = []
    
    for result in search_results:
        content = result["content"]
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                documents.append(chunk)
                metadatas.append({
                    "title": result["title"],
                    "url": result["url"],
                    "topic": topic
                })
                ids.append(str(uuid.uuid4()))
    
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    print(f"✅ RAG Indexing: Stored {len(documents)} chunks in ChromaDB")
    return len(documents)

def retrieve_relevant_chunks(collection, query: str, top_k: int = 5) -> list:
    """RAG RETRIEVAL — Find most relevant chunks using ChromaDB search."""
    count = collection.count()
    if count == 0:
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count)
    )
    
    chunks = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "content": doc,
                "title": results["metadatas"][0][i]["title"],
                "url": results["metadatas"][0][i]["url"],
            })
    
    print(f"✅ RAG Retrieval: Retrieved {len(chunks)} relevant chunks")
    return chunks


def test_rag():
    """Test RAG pipeline."""
    print("Testing RAG pipeline...")
    
    sample_results = [
        {
            "title": "AI Trends 2026",
            "url": "https://example.com/ai-trends",
            "content": "Artificial Intelligence is transforming industries in 2026. Machine learning models are becoming more efficient. Generative AI tools are being adopted widely across healthcare, finance, and education sectors."
        }
    ]
    
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    stored = store_search_results(collection, sample_results, "AI Trends")
    chunks = retrieve_relevant_chunks(collection, "AI in healthcare", top_k=2)
    
    if chunks:
        print(f"✅ RAG working! Retrieved: {chunks[0]['content'][:100]}...")
    else:
        print("❌ RAG retrieval failed")

if __name__ == "__main__":
    test_rag()