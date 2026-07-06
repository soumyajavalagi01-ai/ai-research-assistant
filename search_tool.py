# search_tool.py
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def search_web(query: str, max_results: int = 5) -> list:
    """
    Search the web using Tavily API.
    Returns list of results with title, url, content.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results
    )
    
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", "")
        })
    
    return results


def test_search():
    """Test function to verify Tavily is working."""
    results = search_web("Artificial Intelligence 2026", max_results=2)
    if results:
        print(f"✅ Tavily working! Found {len(results)} results")
        print(f"First result: {results[0]['title']}")
    else:
        print("❌ No results found - check Tavily API key")

if __name__ == "__main__":
    test_search()