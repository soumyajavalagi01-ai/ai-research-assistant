# crew.py
import os
from crewai import Task, Crew, Process
from agents import create_researcher_agent, create_analyst_agent, create_writer_agent
from search_tool import search_web
from rag_engine import (
    get_chroma_client,
    get_or_create_collection,
    store_search_results,
    retrieve_relevant_chunks
)
from dotenv import load_dotenv

load_dotenv()

def run_research(topic: str) -> dict:
    """
    Main pipeline:
    1. Search web (Tavily)
    2. Store in ChromaDB (RAG Indexing)
    3. Retrieve relevant chunks (RAG Retrieval)
    4. Run 3 agents (Researcher → Analyst → Writer)
    5. Return final report
    """
    status_updates = []

    # ── STEP 1: WEB SEARCH ──
    status_updates.append("🔍 Searching the web for latest information...")
    search_results = search_web(topic, max_results=6)
    status_updates.append(f"✅ Found {len(search_results)} web sources")

    # ── STEP 2: RAG INDEXING ──
    status_updates.append("📥 Storing search results in ChromaDB (RAG Indexing)...")
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)
    chunks_stored = store_search_results(collection, search_results, topic)
    status_updates.append(f"✅ Stored {chunks_stored} chunks in vector database")

    # ── STEP 3: RAG RETRIEVAL ──
    status_updates.append("🔎 Retrieving most relevant chunks (Semantic Search)...")
    relevant_chunks = retrieve_relevant_chunks(collection, topic, top_k=6)

    # Format chunks for agents
    context = "\n\n".join([
        f"Source: {c['title']}\nURL: {c['url']}\nContent: {c['content']}"
        for c in relevant_chunks
    ])
    status_updates.append(f"✅ Retrieved {len(relevant_chunks)} relevant chunks via RAG")

    # ── STEP 4: CREATE AGENTS ──
    status_updates.append("🤖 Initializing AI Agents...")
    researcher = create_researcher_agent()
    analyst = create_analyst_agent()
    writer = create_writer_agent()

    # ── STEP 5: CREATE TASKS ──
    research_task = Task(
        description=f"""
        Research the following topic thoroughly: "{topic}"
        
        You have access to the following web-sourced information retrieved via RAG:
        {context}
        
        Your job:
        1. Review all the provided information
        2. Identify the most important and recent facts
        3. Note the key sources
        4. Prepare a comprehensive research summary
        
        Output: A detailed research summary with key facts and sources.
        """,
        agent=researcher,
        expected_output="A comprehensive research summary with key facts and sources about the topic."
    )

    analysis_task = Task(
        description=f"""
        Analyze the research findings about "{topic}".
        
        Based on the researcher's summary:
        1. Extract the top 5-7 key insights
        2. Identify any trends or patterns
        3. Highlight the most important statistics or facts
        4. Organize findings by theme or category
        5. Note any conflicting information or gaps
        
        Output: A structured analysis with key insights and organized findings.
        """,
        agent=analyst,
        expected_output="Structured analysis with key insights, trends, and organized findings."
    )

    writing_task = Task(
        description=f"""
        Write a professional research report on "{topic}".
        
        Use the research and analysis provided by your team.
        Structure the report exactly as follows:

        # Research Report: {topic}

        ## Executive Summary
        (2-3 sentences overview)

        ## Key Findings
        (5-7 bullet points of most important findings)

        ## Detailed Analysis
        (3-4 paragraphs with in-depth discussion)

        ## Trends & Insights
        (Current trends and future outlook)

        ## Conclusion
        (Summary and final thoughts)

        ## Sources
        (List the web sources used)

        Make it professional, clear, and comprehensive.
        """,
        agent=writer,
        expected_output="A complete professional research report with all sections filled."
    )

    # ── STEP 6: RUN CREW ──
    status_updates.append("⚙️ Running Multi-Agent pipeline (Researcher → Analyst → Writer)...")

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    status_updates.append("✅ Research report generated successfully!")

    return {
        "report": str(result),
        "sources": [r["url"] for r in search_results],
        "chunks_stored": chunks_stored,
        "chunks_retrieved": len(relevant_chunks),
        "status_updates": status_updates
    }


def test_crew():
    """Test the full pipeline with a sample topic."""
    print("🚀 Testing full pipeline...")
    print("Topic: Generative AI in 2026")
    print("-" * 50)

    result = run_research("Generative AI in 2026")

    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETE!")
    print(f"Sources found: {len(result['sources'])}")
    print(f"Chunks stored: {result['chunks_stored']}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")
    print("\n📄 REPORT PREVIEW (first 500 chars):")
    print(result["report"][:500])

if __name__ == "__main__":
    test_crew()