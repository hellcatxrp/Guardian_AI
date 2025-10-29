from agents.base_agent import BaseAgent
from shared.knowledge_base import KnowledgeBase
import asyncio
import httpx
from bs4 import BeautifulSoup # New import

class ResearcherAgent(BaseAgent):
    """
    The ResearcherAgent is responsible for gathering information from various sources.
    It performs web searches and fetches content from URLs.
    """
    def __init__(self, config: dict, knowledge_base: KnowledgeBase):
        super().__init__("ResearcherAgent", config, knowledge_base)
        self.brave_search_api_key = config.get("brave_search_api_key")
        self.serper_api_key = config.get("serper_api_key") # New: Serper API key
        if not self.brave_search_api_key and not self.serper_api_key:
            self.logger.warning("No web search API key provided to ResearcherAgent. Web search will be simulated.")
        self.sources = [] # To store gathered sources

    async def _fetch_webpage_content(self, url: str) -> str:
        """Fetches the full text content of a given URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml') # Use lxml parser
                # Extract text from common content areas, ignoring scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()
                text = soup.get_text(separator=' ', strip=True)
                return text
        except httpx.RequestError as e:
            self.logger.error(f"ResearcherAgent: Failed to fetch content from {url}: {e}")
        except Exception as e:
            self.logger.error(f"ResearcherAgent: Error parsing content from {url}: {e}")
        return ""

    async def _perform_brave_search(self, query: str):
        """Performs a web search using the Brave Search API."""
        if not self.brave_search_api_key:
            self.logger.warning("Brave Search API key not available. Cannot perform real Brave search.")
            return []

        headers = {
            "X-Subscription-Token": self.brave_search_api_key
        }
        params = {
            "q": query,
            "count": 5 # Get top 5 results
        }
        url = "https://api.search.brave.com/res/v1/web/search"

        self.logger.info(f"ResearcherAgent: Searching Brave for: '{query}'")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                search_results = []
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"]:
                        full_content = await self._fetch_webpage_content(result.get("url"))
                        search_results.append({
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "snippet": result.get("description"),
                            "content": full_content, # Now includes full content
                            "credibility": 0.7, # Placeholder, could be derived from source reputation
                            "recency": result.get("last_updated") # Brave API might have this
                        })
                self.logger.info(f"ResearcherAgent: Found {len(search_results)} results from Brave Search.")
                return search_results
        except httpx.RequestError as e:
            self.logger.error(f"ResearcherAgent: Brave Search request failed: {e}")
        except httpx.HTTPStatusError as e:
            self.logger.error(f"ResearcherAgent: Brave Search HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            self.logger.error(f"ResearcherAgent: An unexpected error occurred during Brave Search: {e}")
        return []

    async def _perform_serper_search(self, query: str):
        """Performs a web search using the Serper API."""
        if not self.serper_api_key:
            self.logger.warning("Serper API key not available. Cannot perform real Serper search.")
            return []

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        params = {
            "q": query,
            "num": 5 # Get top 5 results
        }
        url = "https://google.serper.dev/search"

        self.logger.info(f"ResearcherAgent: Searching Serper for: '{query}'")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                search_results = []
                if "organic" in data:
                    for result in data["organic"]:
                        full_content = await self._fetch_webpage_content(result.get("link"))
                        search_results.append({
                            "title": result.get("title"),
                            "url": result.get("link"),
                            "snippet": result.get("snippet"),
                            "content": full_content, # Now includes full content
                            "credibility": 0.7, # Placeholder
                            "recency": None # Serper might not provide this directly
                        })
                self.logger.info(f"ResearcherAgent: Found {len(search_results)} results from Serper Search.")
                return search_results
        except httpx.RequestError as e:
            self.logger.error(f"ResearcherAgent: Serper Search request failed: {e}")
        except httpx.HTTPStatusError as e:
            self.logger.error(f"ResearcherAgent: Serper HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            self.logger.error(f"ResearcherAgent: An unexpected error occurred during Serper Search: {e}")
        return []

    def _expand_query(self, original_query: str):
        """
        Expands a query into multiple search variations for better coverage.
        """
        expanded_queries = [original_query]
        
        # Add current date context for news queries
        if any(term in original_query.lower() for term in ['news', 'latest', 'recent', 'current', 'today']):
            import datetime
            current_date = datetime.datetime.now()
            year = current_date.year
            month_name = current_date.strftime("%B")
            
            # Add date-specific queries
            expanded_queries.append(f"{original_query} {year}")
            expanded_queries.append(f"{original_query} {month_name} {year}")
            expanded_queries.append(original_query.replace('latest', 'recent').replace('news', 'developments'))
        
        # Add domain-specific variations for AI queries
        if 'ai' in original_query.lower() or 'artificial intelligence' in original_query.lower():
            expanded_queries.append(original_query.replace('ai', 'artificial intelligence'))
            expanded_queries.append(original_query + ' machine learning')
            expanded_queries.append(original_query + ' technology trends')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in expanded_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries[:3]  # Limit to 3 queries to avoid too many API calls

    def _filter_and_score_sources(self, sources):
        """
        Filters and scores sources based on quality indicators.
        """
        filtered_sources = []
        
        for source in sources:
            # Skip if no content was fetched
            if not source.get('content') or len(source.get('content', '')) < 100:
                continue
            
            # Skip obvious homepage/landing pages
            title = source.get('title', '').lower()
            url = source.get('url', '').lower()
            
            skip_indicators = [
                'home page', 'homepage', 'welcome to', 'about us', 'contact us',
                'privacy policy', 'terms of service', 'sitemap'
            ]
            
            if any(indicator in title for indicator in skip_indicators):
                continue
            
            # Calculate credibility score based on multiple factors
            credibility = 0.5  # Base score
            
            # Domain reputation (simple heuristic)
            reputable_domains = [
                'reuters.com', 'bbc.com', 'cnn.com', 'techcrunch.com', 'theverge.com',
                'arstechnica.com', 'wired.com', 'guardian.com', 'nytimes.com',
                'wsj.com', 'nature.com', 'science.org', 'ieee.org', 'arxiv.org'
            ]
            
            if any(domain in url for domain in reputable_domains):
                credibility += 0.3
            
            # Content quality indicators
            content = source.get('content', '')
            if len(content) > 1000:  # Substantial content
                credibility += 0.1
            if len(content) > 3000:  # Very detailed content
                credibility += 0.1
            
            # Look for quality indicators in content
            quality_indicators = [
                'research', 'study', 'according to', 'data shows', 'report',
                'analysis', 'findings', 'statistics', 'survey', 'published'
            ]
            
            quality_count = sum(1 for indicator in quality_indicators if indicator in content.lower())
            credibility += min(0.2, quality_count * 0.05)  # Max 0.2 bonus
            
            # Penalize very short content
            if len(content) < 300:
                credibility -= 0.2
            
            # Ensure credibility stays within bounds
            credibility = max(0.1, min(1.0, credibility))
            
            # Update source with calculated credibility
            source['credibility'] = credibility
            source['content_length'] = len(content)
            
            filtered_sources.append(source)
        
        # Sort by credibility (highest first) and limit to top sources
        filtered_sources.sort(key=lambda x: x.get('credibility', 0), reverse=True)
        return filtered_sources[:8]  # Return top 8 sources

    async def execute(self, query: str, query_id: str):
        """
        Executes the research task with query expansion and content filtering.
        """
        self.logger.info(f"ResearcherAgent: Starting research for query: '{query}' (ID: {query_id})")
        
        # Expand the query for better coverage
        expanded_queries = self._expand_query(query)
        self.logger.info(f"ResearcherAgent: Expanded query into {len(expanded_queries)} variations")
        
        all_sources = []
        
        # Try each expanded query
        for i, expanded_query in enumerate(expanded_queries):
            if i > 0:  # Add delay between queries to be respectful to APIs
                await asyncio.sleep(1)
            
            query_sources = []
            
            if self.brave_search_api_key:
                query_sources = await self._perform_brave_search(expanded_query)
            
            if not query_sources and self.serper_api_key:
                query_sources = await self._perform_serper_search(expanded_query)
            
            if query_sources:
                all_sources.extend(query_sources)
                self.logger.info(f"ResearcherAgent: Query '{expanded_query}' returned {len(query_sources)} results")
            else:
                self.logger.warning(f"ResearcherAgent: No results for query '{expanded_query}'")
        
        if all_sources:
            # Remove duplicates based on URL
            seen_urls = set()
            unique_sources = []
            for source in all_sources:
                url = source.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            
            # Filter and score sources
            filtered_sources = self._filter_and_score_sources(unique_sources)
            
            # Store sources in knowledge base
            for source in filtered_sources:
                await self.knowledge_base.add_data(query_id, "raw_sources", source)
                self.sources.append(source)
            
            self.logger.info(f"ResearcherAgent: Successfully gathered {len(filtered_sources)} high-quality sources (from {len(all_sources)} total results)")
        else:
            self.logger.warning("No real search results found. Performing simulation.")
            # Simulation fallback remains the same
            await asyncio.sleep(2) # Simulate network delay
            dummy_source_1 = {
                "title": f"Article about {query} from Example.com (Simulated)",
                "url": f"http://example.com/{query.replace(' ', '-')}-simulated",
                "snippet": f"This is some dummy content about {query} from Example.com. It contains various details and keywords related to the topic.",
                "content": f"Full simulated content for {query} from Example.com. This would be a longer text.",
                "credibility": 0.8,
                "recency": "2023-01-15"
            }
            await self.knowledge_base.add_data(query_id, "raw_sources", dummy_source_1)
            self.sources.append(dummy_source_1)
            self.logger.info(f"ResearcherAgent: Found dummy source 1 for '{query}'")

            await asyncio.sleep(1) # Simulate network delay
            dummy_source_2 = {
                "title": f"News report on {query} from NewsSite.org (Simulated)",
                "url": f"http://newssite.org/{query.replace(' ', '-')}-report-simulated",
                "snippet": f"A recent report indicates new findings regarding {query}. This content is from NewsSite.org.",
                "content": f"Full simulated content for {query} from NewsSite.org. This would be a longer text.",
                "credibility": 0.7,
                "recency": "2023-02-20"
            }
            await self.knowledge_base.add_data(query_id, "raw_sources", dummy_source_2)
            self.sources.append(dummy_source_2)
            self.logger.info(f"ResearcherAgent: Found dummy source 2 for '{query}'")

        self.logger.info(f"ResearcherAgent: Finished research for query: '{query}' (ID: {query_id})")
        return True # Indicate success

    async def report_results(self):
        """
        Reports the gathered sources.
        """
        if self.sources:
            self.logger.info(f"ResearcherAgent: Reporting {len(self.sources)} sources found.")
            for source in self.sources:
                self.logger.info(f"  - {source['title']} ({source['url']})")
            return self.sources
        else:
            self.logger.info("ResearcherAgent: No sources found.")
            return []