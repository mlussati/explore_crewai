import warnings
import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Create a search tool
search_tool = SerperDevTool()

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key = os.environ.get("SERPER_API_KEY")

venue_finder = Agent(
    role="Localizador de locais para conferências",
    goal="Encontrar o melhor local para a próxima conferência",
    backstory=(
        "Você é um experiente organizador de eventos com talento para encontrar os locais perfeitos. "
        "Sua experiência garante que todos os requisitos da conferência sejam atendidos de forma eficiente."
        "Seu objetivo é fornecer ao cliente as melhores opções de local possíveis."
    ),
    tools=[search_tool],
    verbose=True
)

venue_quality_assurance_agent = Agent(
    role="Especialista em Garantia de Qualidade de Locais",
    goal="Garantir que os locais selecionados atendam a todos os padrões de qualidade e requisitos do cliente",
    backstory=(
        "Você é meticuloso e atento aos detalhes, garantindo que as opções de locais fornecidas "
        "não sejam apenas adequadas, mas também superem as expectativas do cliente. "
        "Seu trabalho é revisar as opções de locais e fornecer um feedback detalhado."
    ),
    tools=[search_tool],
    verbose=True
)

find_venue_task = Task(
    description=(
        "Realize uma busca completa para encontrar o melhor local para a próxima conferência em São Paulo, Brasil. "
        "Considere fatores como capacidade, localização, comodidades e preços. "
        "Use recursos e bases de dados online para reunir informações completas."
    ),
    expected_output=(
        "Uma lista com 5 locais em potencial, contendo informações detalhadas sobre capacidade, localização, comodidades, preços e disponibilidade."
    ),
    tools=[search_tool],
    agent=venue_finder,
)

quality_assurance_review_task = Task(
    description=(
        "Revise as opções de locais fornecidas pelo Localizador de Locais para Conferências. "
        "Garanta que cada local atenda a todos os requisitos e padrões especificados. "
        "Forneça um relatório detalhado sobre a adequação de cada local."
    ),
    expected_output=(
        "Uma análise detalhada dos 5 locais em potencial, destacando quaisquer problemas, pontos fortes e a adequação geral."
    ),
    tools=[search_tool],
    agent=venue_quality_assurance_agent,
)

event_planning_crew = Crew(
    agents=[venue_finder, venue_quality_assurance_agent],
    tasks=[find_venue_task, quality_assurance_review_task],
    verbose=True
)

inputs = {
    "conference_name": "Inovações em IA PalancaCode Summit",
    "requirements": "Capacidade para 5000 pessoas, localização central, comodidades modernas, orçamento de até 292.117,50 reais"
}

result = event_planning_crew.kickoff(inputs=inputs)