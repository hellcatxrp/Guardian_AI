from agents.base_agent import BaseAgent
from shared.knowledge_base import KnowledgeBase
import asyncio

class AnalystAgent(BaseAgent):
    """
    The AnalystAgent processes raw data from researchers, extracts key insights,
    summarizes content, and performs semantic clustering.
    """
    def __init__(self, config: dict, knowledge_base: KnowledgeBase):
        super().__init__("AnalystAgent", config, knowledge_base)
        self.llm_model = config.get("llm_model")
        if not self.llm_model:
            self.logger.warning("LLM model not provided to AnalystAgent. Analysis will be simulated.")
        self.insights = [] # To store extracted insights

    async def execute(self, query_id: str):
        """
        Executes the analysis task by retrieving raw sources from the KnowledgeBase,
        simulating analysis, and adding processed data back.
        """
        self.logger.info(f"AnalystAgent: Starting analysis for query ID: {query_id}")
        
        raw_sources = await self.knowledge_base.get_data(query_id, "raw_sources")
        if not raw_sources:
            self.logger.warning(f"AnalystAgent: No raw sources found for query ID: {query_id}. Skipping analysis.")
            return False

        processed_content = []
        for i, source in enumerate(raw_sources):
            self.logger.info(f"AnalystAgent: Analyzing source: {source.get('title', source.get('url'))}")
            
            content_to_analyze = source.get('content', source.get('snippet', ''))
            if not content_to_analyze:
                self.logger.warning(f"AnalystAgent: No content to analyze for source: {source.get('title', source.get('url'))}")
                continue

            if self.llm_model:
                try:
                    # Chunk content for better processing (max 3000 chars for safety)
                    content_chunk = content_to_analyze[:3000] if len(content_to_analyze) > 3000 else content_to_analyze
                    
                    # Enhanced prompt with structured output request
                    prompt = (
                        f"Analyze the following content and provide a structured response.\n"
                        f"Source: {source.get('title', 'Unknown')}\n"
                        f"URL: {source.get('url', 'Unknown')}\n\n"
                        f"Please provide:\n"
                        f"1. A concise 2-3 sentence summary of the main topic\n"
                        f"2. 3-5 key insights or facts (as bullet points)\n"
                        f"3. Any important dates, numbers, or statistics mentioned\n\n"
                        f"Content:\n{content_chunk}\n\n"
                        f"Format your response as:\n"
                        f"SUMMARY: [your summary here]\n"
                        f"KEY_INSIGHTS:\n"
                        f"- [insight 1]\n"
                        f"- [insight 2]\n"
                        f"- [etc.]\n"
                        f"STATISTICS: [any relevant numbers/dates]"
                    )
                    
                    # Generate content with retry logic
                    response = None
                    for attempt in range(2):  # Try twice
                        try:
                            response = self.llm_model.generate_content(prompt)
                            if response and hasattr(response, 'text') and response.text:
                                break
                        except Exception as retry_error:
                            self.logger.warning(f"LLM attempt {attempt + 1} failed: {retry_error}")
                            if attempt == 1:  # Last attempt
                                raise retry_error
                            await asyncio.sleep(1)  # Wait before retry
                    
                    if not response or not hasattr(response, 'text') or not response.text:
                        raise Exception("Empty response from LLM")
                    
                    llm_output = response.text.strip()
                    
                    # Parse structured output
                    summary = "No summary available"
                    key_points = []
                    statistics = ""
                    
                    current_section = None
                    for line in llm_output.split('\n'):
                        line = line.strip()
                        if line.startswith("SUMMARY:"):
                            summary = line[8:].strip()
                            current_section = "summary"
                        elif line.startswith("KEY_INSIGHTS:"):
                            current_section = "insights"
                        elif line.startswith("STATISTICS:"):
                            statistics = line[11:].strip()
                            current_section = "stats"
                        elif line.startswith("- ") and current_section == "insights":
                            key_points.append(line[2:].strip())
                        elif current_section == "summary" and line and not line.startswith(("KEY_INSIGHTS:", "STATISTICS:")):
                            summary += " " + line
                    
                    # Fallback parsing if structured format wasn't followed
                    if summary == "No summary available" and llm_output:
                        lines = llm_output.split('\n')
                        summary = lines[0][:200] + "..." if len(lines[0]) > 200 else lines[0]
                    
                    if not key_points and llm_output:
                        # Extract bullet points from anywhere in the response
                        for line in llm_output.split('\n'):
                            if line.strip().startswith('- '):
                                key_points.append(line.strip()[2:])
                        
                        # If still no points, create one from summary
                        if not key_points:
                            key_points = [summary[:100] + "..." if len(summary) > 100 else summary]

                except Exception as e:
                    self.logger.error(f"AnalystAgent: LLM analysis failed for source {source.get('url')}: {e}")
                    # Better fallback with actual content snippet
                    summary = f"Failed to analyze content from {source.get('title', 'Unknown source')}. Content preview: {content_to_analyze[:150]}..."
                    key_points = [f"Content analysis failed due to: {str(e)[:100]}"]
            else:
                # Fallback to simulated analysis if LLM not available
                summary = f"Summary of '{source['title']}': This article discusses {content_to_analyze[:50]}... (simulated summary)"
                key_points = [f"Simulated Point A from source {i+1}", f"Simulated Point B from source {i+1}"]

            insight = {
                "source_url": source['url'],
                "title": source.get('title'),
                "summary": summary,
                "key_points": key_points,
                "original_snippet": source.get('snippet')
            }
            self.insights.append(insight)
            self.logger.info(f"AnalystAgent: Processed source: {source.get('title', source.get('url'))}")

        # Overall insight generation using LLM
        if self.llm_model and self.insights:
            try:
                # Compile all insights for overall analysis
                all_summaries = []
                all_key_points = []
                for insight in self.insights:
                    if insight.get('summary'):
                        all_summaries.append(insight['summary'])
                    if insight.get('key_points'):
                        all_key_points.extend(insight['key_points'])
                
                # Create prompt for overall analysis
                combined_content = "\n".join(all_summaries[:5])  # Limit to first 5 summaries
                combined_points = "\n".join([f"- {point}" for point in all_key_points[:10]])  # Limit to 10 points
                
                overall_prompt = (
                    f"Analyze the following research summaries and key points to provide an overall assessment.\n\n"
                    f"SUMMARIES:\n{combined_content}\n\n"
                    f"KEY POINTS:\n{combined_points}\n\n"
                    f"Please provide:\n"
                    f"1. An overall synthesis of the main themes (2-3 sentences)\n"
                    f"2. Any contradictions or conflicting information found\n"
                    f"3. Confidence level in the information (0.1-1.0)\n\n"
                    f"Format as:\n"
                    f"SYNTHESIS: [your synthesis]\n"
                    f"CONTRADICTIONS: [any contradictions found or 'None detected']\n"
                    f"CONFIDENCE: [0.1-1.0]"
                )
                
                response = self.llm_model.generate_content(overall_prompt)
                if response and hasattr(response, 'text') and response.text:
                    analysis_output = response.text.strip()
                    
                    # Parse the structured output
                    synthesis = "Overall analysis completed"
                    contradictions = "None detected"
                    confidence = 0.75
                    
                    for line in analysis_output.split('\n'):
                        line = line.strip()
                        if line.startswith("SYNTHESIS:"):
                            synthesis = line[9:].strip()
                        elif line.startswith("CONTRADICTIONS:"):
                            contradictions = line[15:].strip()
                        elif line.startswith("CONFIDENCE:"):
                            try:
                                confidence = float(line[11:].strip())
                            except:
                                confidence = 0.75
                    
                    overall_insight = {
                        "type": "overall_analysis",
                        "summary": synthesis,
                        "contradictions_detected": contradictions != "None detected",
                        "contradictions": contradictions,
                        "confidence_score": confidence
                    }
                else:
                    raise Exception("Empty response from LLM")
                    
            except Exception as e:
                self.logger.error(f"AnalystAgent: Overall analysis failed: {e}")
                overall_insight = {
                    "type": "overall_analysis",
                    "summary": f"Overall analysis of {len(self.insights)} sources completed. LLM analysis failed: {str(e)[:100]}",
                    "contradictions_detected": False,
                    "confidence_score": 0.5
                }
        else:
            overall_insight = {
                "type": "overall_analysis",
                "summary": f"Overall analysis of {len(self.insights)} sources completed (no LLM available)",
                "contradictions_detected": False,
                "confidence_score": 0.3
            }
        
        self.insights.append(overall_insight)
        # Add each insight individually to the knowledge base
        for insight_item in self.insights:
            await self.knowledge_base.add_data(query_id, "analyzed_data", insight_item)
        
        self.logger.info(f"AnalystAgent: Finished analysis for query ID: {query_id}. Extracted {len(self.insights)} insights.")
        return True

    async def report_results(self):
        """
        Reports the extracted insights.
        """
        if self.insights:
            self.logger.info(f"AnalystAgent: Reporting {len(self.insights)} insights found.")
            for insight in self.insights:
                self.logger.info(f"  - Insight: {insight.get('summary', insight.get('type'))}")
            return self.insights
        else:
            self.logger.info("AnalystAgent: No insights found.")
            return []
