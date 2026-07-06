# agents.py
import os
from crewai import Agent
from groq import Groq
from langchain.llms.base import LLM
from typing import Optional, List, Any
from dotenv import load_dotenv

load_dotenv()

class GroqLLM(LLM):
    """Custom Groq LLM wrapper for CrewAI."""
    model_name: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        client = Groq(api_key=self.groq_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048
        )
        return response.choices[0].message.content

def get_llm():
    return GroqLLM(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

def create_researcher_agent():
    return Agent(
        role="Senior Research Specialist",
        goal="Search and gather comprehensive accurate information on the given topic.",
        backstory="""You are an expert researcher with 10+ years of experience 
        gathering information. You find the most relevant and reliable sources.""",
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

def create_analyst_agent():
    return Agent(
        role="Senior Data Analyst",
        goal="Analyze retrieved information and identify key insights and patterns.",
        backstory="""You are an analytical expert who extracts the most important 
        insights from large amounts of information and organizes them logically.""",
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

def create_writer_agent():
    return Agent(
        role="Professional Research Report Writer",
        goal="Write a clear comprehensive well-structured research report.",
        backstory="""You are a professional writer specializing in clear engaging 
        research reports with executive summary, key findings, and conclusion.""",
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

def test_agents():
    print("Testing agent creation...")
    try:
        researcher = create_researcher_agent()
        print(f"✅ Researcher Agent: {researcher.role}")
        analyst = create_analyst_agent()
        print(f"✅ Analyst Agent: {analyst.role}")
        writer = create_writer_agent()
        print(f"✅ Writer Agent: {writer.role}")
        print("\n✅ All 3 agents created successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_agents()