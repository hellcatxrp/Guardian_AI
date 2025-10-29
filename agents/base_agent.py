import abc
import logging

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the multi-agent research system.
    Defines common functionalities like logging, configuration, and standardized interfaces.
    """

    def __init__(self, name: str, config: dict, knowledge_base):
        self.name = name
        self.config = config
        self.knowledge_base = knowledge_base
        self._setup_logging()

    def _setup_logging(self):
        """Sets up a logger for the agent."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @abc.abstractmethod
    async def execute(self, query: str):
        """
        Abstract method to execute the agent's primary task.
        Must be implemented by concrete agent classes.
        """
        pass

    @abc.abstractmethod
    async def report_results(self):
        """
        Abstract method to report the agent's findings or results.
        Must be implemented by concrete agent classes.
        """
        pass

    async def cleanup(self):
        """
        Performs any necessary cleanup before the agent shuts down.
        Can be overridden by concrete agent classes.
        """
        self.logger.info(f"{self.name} is performing cleanup.")