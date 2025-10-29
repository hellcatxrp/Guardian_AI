import asyncio
from collections import defaultdict

class KnowledgeBase:
    """
    A centralized, thread-safe (using asyncio locks) knowledge base for agents to share and retrieve data.
    Stores sources, insights, and validation results, organized by query or session.
    """
    def __init__(self):
        self._data = defaultdict(lambda: defaultdict(list)) # query_id -> category -> list of items
        self._locks = defaultdict(asyncio.Lock) # Lock per query_id for thread safety

    async def add_data(self, query_id: str, category: str, item):
        """
        Adds an item to the knowledge base under a specific query_id and category.
        """
        async with self._locks[query_id]:
            self._data[query_id][category].append(item)
            # self.logger.info(f"Added data to KB for query {query_id}, category {category}") # Add logging later

    async def get_data(self, query_id: str, category: str = None):
        """
        Retrieves data from the knowledge base for a specific query_id and optional category.
        If category is None, returns all data for the query_id.
        """
        async with self._locks[query_id]:
            if category:
                return self._data[query_id].get(category, [])
            return self._data[query_id]

    async def clear_query_data(self, query_id: str):
        """
        Clears all data associated with a specific query_id.
        """
        async with self._locks[query_id]:
            if query_id in self._data:
                del self._data[query_id]
            if query_id in self._locks:
                del self._locks[query_id]
            # self.logger.info(f"Cleared data for query {query_id} from KB.") # Add logging later