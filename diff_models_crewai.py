import warnings
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# Create a search tool
search_tool = SerperDevTool()

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key = os.environ.get("SERPER_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Call the gemini models
gemini= ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                               verbose=True,
                               temperature=0.5,
                               google_api_key=google_api_key,)
# Initialize the GPT-4 model using CHatOpenAI
gpt=ChatOpenAI(model="gpt-4o-2024-08-06",
               verbose=True,
               temperature=0.5,
               openai_api_key=openai_api_key)

# Data Researcher Agent using Gemini and SerperSearch
article_researcher = Agent(
    role="Pesquisador Senior",
    goal='Descubra tecnologias inovadoras em {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Motivado pela curiosidade, você est;a na vanguarda da"
        "inovação, ansioso por explorar e compartilhar conhecimento que podem mudar"
        "o mundo."
    ),
    tools=[search_tool],
    llm=gemini,
    allow_delegation=True
)

# Article Writer Agent using GPT
#Todo: Add a tool to the article writer agent