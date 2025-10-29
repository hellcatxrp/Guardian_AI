import asyncio
import uuid
import logging

from shared.knowledge_base import KnowledgeBase
from agents.researcher_agent import ResearcherAgent
from agents.analyst_agent import AnalystAgent
from agents.critic_agent import CriticAgent
from agents.synthesizer_agent import SynthesizerAgent

# Configure logging for the orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    """
    The Orchestrator manages the lifecycle of research agents, coordinates their execution
    through defined phases, and provides real-time updates.
    """
    def __init__(self, config: dict = None):
        self.config = config if config is not None else {}
        self.knowledge_base = KnowledgeBase()
        
        # Initialize agents
        self.researcher = ResearcherAgent(self.config.get("researcher", {}), self.knowledge_base)
        self.analyst = AnalystAgent(self.config.get("analyst", {}), self.knowledge_base)
        self.critic = CriticAgent(self.config.get("critic", {}), self.knowledge_base)
        self.synthesizer = SynthesizerAgent(self.config.get("synthesizer", {}), self.knowledge_base)

    async def execute_research(self, query: str) -> str:
        """
        Coordinates the execution of the research agents through the defined phases.
        """
        query_id = str(uuid.uuid4())
        logger.info(f"Orchestrator: Starting research for query: '{query}' with ID: {query_id}")

        try:
            # Phase 1: Gathering
            logger.info(f"Orchestrator: Phase 1 - Gathering (ResearcherAgent)")
            research_success = await self.researcher.execute(query, query_id)
            if not research_success:
                logger.error(f"Orchestrator: ResearcherAgent failed for query ID: {query_id}")
                return "Research failed during gathering phase."

            # Phase 2: Analysis
            logger.info(f"Orchestrator: Phase 2 - Analysis (AnalystAgent)")
            analysis_success = await self.analyst.execute(query_id)
            if not analysis_success:
                logger.error(f"Orchestrator: AnalystAgent failed for query ID: {query_id}")
                return "Research failed during analysis phase."

            # Phase 3: Validation
            logger.info(f"Orchestrator: Phase 3 - Validation (CriticAgent)")
            critic_success = await self.critic.execute(query_id)
            if not critic_success:
                logger.error(f"Orchestrator: CriticAgent failed for query ID: {query_id}")
                return "Research failed during validation phase."

            # Phase 4: Synthesis
            logger.info(f"Orchestrator: Phase 4 - Synthesis (SynthesizerAgent)")
            synthesis_success = await self.synthesizer.execute(query, query_id)
            if not synthesis_success:
                logger.error(f"Orchestrator: SynthesizerAgent failed for query ID: {query_id}")
                return "Research failed during synthesis phase."

            final_response_list = await self.knowledge_base.get_data(query_id, "final_response")
            final_response = final_response_list[0] if final_response_list else "No final response generated."
            logger.info(f"Orchestrator: Research completed for query ID: {query_id}")
            return final_response

        except Exception as e:
            logger.info(f"Orchestrator: An unexpected error occurred during research for query ID: {query_id}")
            return f"An unexpected error occurred: {str(e)}"
        finally:
            # Clean up knowledge base data for this query
            await self.knowledge_base.clear_query_data(query_id)
            logger.info(f"Orchestrator: Cleaned up data for query ID: {query_id}")

async def main():
    orchestrator = Orchestrator()
    
    test_query = "The impact of AI on future job markets"
    print(f"\n--- Running research for: '{test_query}' ---")
    response = await orchestrator.execute_research(test_query)
    print("\n--- Final Research Report ---")
    print(response)
    print("-----------------------------")

    test_query_2 = "Recent advancements in quantum computing"
    print(f"\n--- Running research for: '{test_query_2}' ---")
    response_2 = await orchestrator.execute_research(test_query_2)
    print("\n--- Final Research Report ---")
    print(response_2)
    print("-----------------------------")

if __name__ == "__main__":
    asyncio.run(main())
