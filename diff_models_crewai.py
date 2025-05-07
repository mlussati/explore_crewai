import warnings
import os

print("[LOG] Importando bibliotecas...")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

print("[LOG] Carregando ferramentas de busca...")
search_tool = SerperDevTool()

print("[LOG] Carregando variáveis de ambiente...")
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key = os.environ.get("SERPER_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

print("[LOG] Inicializando modelos LLM...")

# Função para inicializar o modelo Google com o provider corretamente configurado
def create_google_model():
    print("[LOG] Configurando o modelo Google Gemini...")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-001", 
        verbose=True, 
        temperature=0.5,
        google_api_key=google_api_key
    )
    
    # Verificando e garantindo que o provider seja corretamente configurado
    if not hasattr(model, "provider") or model.provider is None:
        print("[ALERTA] Provider não está configurado. Configurando manualmente como 'google'.")
        model._llm_provider = "google"  # Forçando o provider internamente

    return model

# Inicializando o modelo Gemini com o provider explicitamente definido como Google
gemini = create_google_model()

# Initialize the GPT-4 model using CHatOpenAI
gpt = ChatOpenAI(
    model="gpt-4-0125-preview",
    verbose=True, 
    temperature=0.5, 
    openai_api_key=openai_api_key
)

print("[LOG] Configurando agentes...")
# Data Researcher Agent using Gemini and SerperSearch
article_researcher = Agent(
    role="Pesquisador Senior",
    goal='Descubra tecnologias inovadoras em {topic}',
    verbose=True,
    memory=True,
    backstory="Motivado pela curiosidade, você está na vanguarda da inovação.",
    tools=[search_tool],
    llm=gemini,
    allow_delegation=True
)

# Article Writer Agent using GPT
article_writer = Agent(
    role="Escritor de Artigos",
    goal="Narre histórias tecnológicas envolventes sobre {topic}",
    verbose=True,
    memory=True,
    backstory="Com talento para simplificar tópicos complexos, você cria narrativas envolventes.",
    tools=[search_tool],
    llm=gpt,
    allow_delegation=True
)

print("[LOG] Definindo tarefas...")
# Definindo Tasks
research_task = Task(
    description="Realize uma análise completa sobre o tópico em questão.",
    expected_output='Um relatório detalhado sobre a análise de dados com insights importantes',
    tools=[search_tool],
    agent=article_researcher,
)

writing_task = Task(
    description="Escreva um artigo perspicaz com base no relatório de análise de dados.",
    expected_output='Um artigo de 6 parágrafos resumindo os insights dos dados.',
    agent=article_writer,
)

print("[LOG] Configurando o Crew...")
# Form the crew and define the process
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

research_inputs = {'topic': 'O aumento das temperaturas globais a partir de 2018'}

print("[LOG] Iniciando o processo com o Crew...")
result = crew.kickoff(inputs=research_inputs)

print("[LOG] Processo concluído. Resultados:")
print(result)
