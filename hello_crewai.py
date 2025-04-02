from crewai import Agent
from crewai import Task
from crewai_tools import SerperDevTool


# Create a search tool
search_tool = SerperDevTool()

# Define agents
venue_finder = Agent(
    role='Localizador de Locais para Conferências',
    goal='Encontre o melhor local para a próxima conferência',
    backstory="Você é um planejador de eventos experiente, com um talento especial para encontrar os locais perfeitos. \
                Sua experiência garante que todos os requisitos da conferência sejam atendidos de forma eficiente.",
    verbose=True,
    tools=[search_tool]
)

# Define tasks
find_venue_task = Task(
    description=(
        "Realize uma busca completa para encontrar o melhor local para a próxima "
        "conferência. Considere fatores como capacidade, localização, comodidades "
        "e preços. Use recursos online e bancos de dados para reunir informações "
        "abrangentes."
    ),
    expected_output=(
        "Uma lista com 5 locais potenciais, contendo informações detalhadas sobre "
        "capacidade, localização, comodidades, preços e disponibilidade."
    ),
    agent=venue_finder
)

