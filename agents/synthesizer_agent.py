from agents.base_agent import BaseAgent
from shared.knowledge_base import KnowledgeBase
import asyncio

class SynthesizerAgent(BaseAgent):
    """
    The SynthesizerAgent compiles the final intelligent response, integrating
    multiple perspectives and managing citations.
    """
    def __init__(self, config: dict, knowledge_base: KnowledgeBase):
        super().__init__("SynthesizerAgent", config, knowledge_base)
        self.llm_model = config.get("llm_model")
        if not self.llm_model:
            self.logger.warning("LLM model not provided to SynthesizerAgent. Synthesis will be basic.")
        self.final_response = None

    async def execute(self, query: str, query_id: str):
        """
        Executes the synthesis task by retrieving validated data from the KnowledgeBase,
        simulating response compilation, and storing the final response.
        """
        self.logger.info(f"SynthesizerAgent: Starting synthesis for query: '{query}' (ID: {query_id})")
        
        validated_data = await self.knowledge_base.get_data(query_id, "validated_data")
        analyzed_data = await self.knowledge_base.get_data(query_id, "analyzed_data")
        raw_sources = await self.knowledge_base.get_data(query_id, "raw_sources")

        if not validated_data and not analyzed_data:
            self.logger.warning(f"SynthesizerAgent: No validated or analyzed data found for query ID: {query_id}. Cannot synthesize response.")
            return False

        # Intelligent synthesis using LLM if available
        if self.llm_model and (analyzed_data or validated_data):
            try:
                # Prepare content for synthesis
                key_insights = []
                validation_summary = ""
                sources_info = []
                
                # Collect key insights
                for insight in analyzed_data:
                    if insight.get('type') != 'overall_analysis':
                        summary = insight.get('summary', 'No summary')
                        key_points = insight.get('key_points', [])
                        source_title = insight.get('title', 'Unknown source')
                        key_insights.append({
                            'summary': summary,
                            'points': key_points,
                            'source': source_title
                        })
                    elif insight.get('type') == 'overall_analysis':
                        # Get overall analysis from analyst
                        overall_analysis = insight.get('summary', '')
                
                # Get validation information
                overall_validation = next((item for item in validated_data if item.get('type') == 'overall_validation'), None)
                if overall_validation:
                    validation_summary = overall_validation.get('summary', '')
                    validation_confidence = overall_validation.get('overall_confidence', 0)
                    gaps = overall_validation.get('gaps_identified', [])
                
                # Prepare sources
                for source in raw_sources[:5]:  # Limit to top 5 sources
                    sources_info.append({
                        'title': source.get('title', 'Unknown'),
                        'url': source.get('url', ''),
                        'credibility': source.get('credibility', 0.5)
                    })
                
                # Create comprehensive synthesis prompt
                synthesis_prompt = (
                    f"Create a comprehensive research report for the query: '{query}'\n\n"
                    f"Using the following analyzed data:\n\n"
                )
                
                # Add insights to prompt
                for i, insight in enumerate(key_insights[:5]):  # Limit to 5 insights
                    synthesis_prompt += f"INSIGHT {i+1} (from {insight['source']}):\n"
                    synthesis_prompt += f"Summary: {insight['summary'][:300]}...\n"
                    if insight['points']:
                        synthesis_prompt += f"Key Points: {'; '.join(insight['points'][:3])}\n"
                    synthesis_prompt += "\n"
                
                if validation_summary:
                    synthesis_prompt += f"VALIDATION: {validation_summary}\n\n"
                
                synthesis_prompt += (
                    f"Create a well-structured report with:\n"
                    f"1. Executive Summary (2-3 sentences)\n"
                    f"2. Key Findings (3-5 main points with specifics)\n"
                    f"3. Important Details (dates, numbers, key facts)\n"
                    f"4. Conclusions and Implications\n"
                    f"5. Confidence Assessment\n\n"
                    f"Format as:\n"
                    f"# Research Report: {query}\n\n"
                    f"## Executive Summary\n"
                    f"[2-3 sentence overview]\n\n"
                    f"## Key Findings\n"
                    f"[Numbered list of main findings]\n\n"
                    f"## Important Details\n"
                    f"[Specific facts, dates, numbers]\n\n"
                    f"## Conclusions\n"
                    f"[Implications and takeaways]\n\n"
                    f"## Confidence Assessment\n"
                    f"[Overall confidence level and any caveats]"
                )
                
                # Generate synthesis
                response = self.llm_model.generate_content(synthesis_prompt)
                if response and hasattr(response, 'text') and response.text:
                    synthesized_content = response.text.strip()
                    
                    # Add sources section
                    sources_section = "\n\n## Sources\n"
                    for source in sources_info:
                        credibility_label = "High" if source['credibility'] > 0.8 else "Medium" if source['credibility'] > 0.6 else "Standard"
                        sources_section += f"- {source['title']} ([Link]({source['url']})) - {credibility_label} Credibility\n"
                    
                    self.final_response = synthesized_content + sources_section
                    
                else:
                    raise Exception("Empty response from synthesis LLM")
                    
            except Exception as e:
                self.logger.error(f"SynthesizerAgent: LLM synthesis failed: {e}")
                # Fallback to structured template
                await self._create_structured_fallback_response(query, analyzed_data, validated_data, raw_sources)
        else:
            # Fallback when no LLM available
            await self._create_structured_fallback_response(query, analyzed_data, validated_data, raw_sources)
        await self.knowledge_base.add_data(query_id, "final_response", self.final_response)
        
        self.logger.info(f"SynthesizerAgent: Finished synthesis for query: '{query}' (ID: {query_id})")
        return True

    async def _create_structured_fallback_response(self, query: str, analyzed_data, validated_data, raw_sources):
        """
        Creates a structured fallback response when LLM is not available or fails.
        """
        response_parts = [f"# Research Report: {query}\n\n"]
        
        # Executive Summary
        response_parts.append("## Executive Summary\n")
        if analyzed_data:
            overall_analysis = next((item for item in analyzed_data if item.get('type') == 'overall_analysis'), None)
            if overall_analysis:
                response_parts.append(f"{overall_analysis.get('summary', 'Research completed across multiple sources.')}\n\n")
            else:
                response_parts.append(f"Research analysis completed across {len(analyzed_data)} sources providing insights on {query}.\n\n")
        else:
            response_parts.append("Research completed with basic data gathering.\n\n")
        
        # Key Findings
        if analyzed_data:
            response_parts.append("## Key Findings\n")
            finding_count = 1
            for insight in analyzed_data:
                if insight.get('type') != 'overall_analysis' and finding_count <= 5:
                    response_parts.append(f"{finding_count}. {insight.get('summary', 'Finding available')}\n")
                    if insight.get('key_points'):
                        for point in insight['key_points'][:2]:  # Limit to 2 points per insight
                            response_parts.append(f"   - {point}\n")
                    finding_count += 1
            response_parts.append("\n")

        # Validation Summary
        if validated_data:
            response_parts.append("## Validation Summary\n")
            overall_validation = next((item for item in validated_data if item.get('type') == 'overall_validation'), None)
            if overall_validation:
                response_parts.append(f"**Validation Status:** {overall_validation.get('summary', 'Validation completed')}\n")
                if overall_validation.get('gaps_identified'):
                    response_parts.append(f"**Identified Gaps:** {', '.join(overall_validation['gaps_identified'])}\n")
                response_parts.append(f"**Overall Confidence:** {overall_validation.get('overall_confidence', 0):.2f}/1.0\n\n")

        # Sources
        if raw_sources:
            response_parts.append("## Sources\n")
            for i, source in enumerate(raw_sources[:5], 1):
                credibility = source.get('credibility', 0.5)
                credibility_label = "High" if credibility > 0.8 else "Medium" if credibility > 0.6 else "Standard"
                response_parts.append(f"{i}. {source['title']} - {credibility_label} Credibility\n")
                response_parts.append(f"   Link: {source['url']}\n")
            response_parts.append("\n")

        self.final_response = "".join(response_parts)

    async def report_results(self):
        """
        Reports the final synthesized response.
        """
        if self.final_response:
            self.logger.info("SynthesizerAgent: Reporting final synthesized response.")
            return self.final_response
        else:
            self.logger.info("SynthesizerAgent: No final response synthesized.")
            return None