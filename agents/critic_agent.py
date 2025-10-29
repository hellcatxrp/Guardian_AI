from agents.base_agent import BaseAgent
from shared.knowledge_base import KnowledgeBase
import asyncio

class CriticAgent(BaseAgent):
    """
    The CriticAgent performs multi-source fact verification, bias detection,
    and source credibility assessment.
    """
    def __init__(self, config: dict, knowledge_base: KnowledgeBase):
        super().__init__("CriticAgent", config, knowledge_base)
        self.llm_model = config.get("llm_model")
        if not self.llm_model:
            self.logger.warning("LLM model not provided to CriticAgent. Validation will be simulated.")
        self.validations = [] # To store validation results

    async def execute(self, query_id: str):
        """
        Executes the critical assessment task by retrieving analyzed data from the KnowledgeBase,
        simulating validation, and adding results back.
        """
        self.logger.info(f"CriticAgent: Starting critical assessment for query ID: {query_id}")
        
        analyzed_data = await self.knowledge_base.get_data(query_id, "analyzed_data")
        if not analyzed_data:
            self.logger.warning(f"CriticAgent: No analyzed data found for query ID: {query_id}. Skipping criticism.")
            return False

        for i, insight in enumerate(analyzed_data):
            if insight.get('type') == 'overall_analysis':
                continue  # Skip overall analysis, we'll handle it separately
                
            insight_summary = insight.get('summary', insight.get('type', 'Unknown insight'))
            
            if self.llm_model:
                try:
                    # Create validation prompt
                    validation_prompt = (
                        f"Please critically evaluate the following research insight for accuracy, bias, and reliability.\n\n"
                        f"INSIGHT: {insight_summary}\n"
                        f"SOURCE: {insight.get('title', 'Unknown')}\n"
                        f"KEY POINTS: {', '.join(insight.get('key_points', []))}\n\n"
                        f"Evaluate:\n"
                        f"1. Factual accuracy (any obvious errors or inconsistencies?)\n"
                        f"2. Potential bias (language, perspective, missing viewpoints?)\n"
                        f"3. Source reliability (based on content quality and presentation)\n"
                        f"4. Recency/relevance of information\n\n"
                        f"Format your response as:\n"
                        f"ACCURACY: [High/Medium/Low - with brief explanation]\n"
                        f"BIAS: [None/Low/Medium/High - with brief explanation]\n"
                        f"RELIABILITY: [0.1-1.0 score]\n"
                        f"ISSUES: [any specific concerns or 'None identified']"
                    )
                    
                    response = self.llm_model.generate_content(validation_prompt)
                    if response and hasattr(response, 'text') and response.text:
                        validation_output = response.text.strip()
                        
                        # Parse validation response
                        accuracy = "Medium"
                        bias_level = "None"
                        reliability_score = 0.75
                        issues = "None identified"
                        
                        for line in validation_output.split('\n'):
                            line = line.strip()
                            if line.startswith("ACCURACY:"):
                                accuracy = line[9:].strip()
                            elif line.startswith("BIAS:"):
                                bias_level = line[5:].strip()
                            elif line.startswith("RELIABILITY:"):
                                try:
                                    reliability_score = float(line[12:].strip())
                                except:
                                    reliability_score = 0.75
                            elif line.startswith("ISSUES:"):
                                issues = line[7:].strip()
                        
                        # Convert to structured format
                        bias_detected = bias_level.lower() not in ['none', 'low']
                        fact_checked = accuracy.lower() in ['high', 'medium']
                        
                        validation_result = {
                            "insight_summary": insight_summary,
                            "fact_checked": fact_checked,
                            "accuracy_level": accuracy,
                            "bias_detected": bias_detected,
                            "bias_level": bias_level,
                            "credibility_score": reliability_score,
                            "confidence_score": reliability_score * 0.9,  # Slightly lower than reliability
                            "issues_identified": issues,
                            "validation_method": "LLM_analysis"
                        }
                    else:
                        raise Exception("Empty response from LLM")
                        
                except Exception as e:
                    self.logger.error(f"CriticAgent: LLM validation failed for insight: {e}")
                    # Fallback validation
                    validation_result = {
                        "insight_summary": insight_summary,
                        "fact_checked": False,
                        "accuracy_level": "Unknown",
                        "bias_detected": False,
                        "credibility_score": 0.5,
                        "confidence_score": 0.4,
                        "issues_identified": f"Validation failed: {str(e)[:100]}",
                        "validation_method": "fallback"
                    }
            else:
                # Simple heuristic validation when no LLM available
                content_length = len(insight_summary)
                has_numbers = any(char.isdigit() for char in insight_summary)
                has_specific_terms = any(term in insight_summary.lower() for term in ['study', 'research', 'according to', 'reported'])
                
                # Simple scoring based on content characteristics
                credibility = 0.6
                if has_numbers:
                    credibility += 0.1
                if has_specific_terms:
                    credibility += 0.1
                if content_length > 100:
                    credibility += 0.1
                if content_length < 50:
                    credibility -= 0.1
                    
                credibility = max(0.1, min(1.0, credibility))
                
                validation_result = {
                    "insight_summary": insight_summary,
                    "fact_checked": True,
                    "accuracy_level": "Medium",
                    "bias_detected": False,
                    "credibility_score": credibility,
                    "confidence_score": credibility * 0.8,
                    "issues_identified": "Limited validation (no LLM available)",
                    "validation_method": "heuristic"
                }
            
            self.validations.append(validation_result)
            self.logger.info(f"CriticAgent: Validated insight: {validation_result['insight_summary'][:100]}...")

        # Calculate overall validation metrics
        individual_confidence_scores = [v['confidence_score'] for v in self.validations if 'confidence_score' in v]
        calculated_overall_confidence = sum(individual_confidence_scores) / len(individual_confidence_scores) if individual_confidence_scores else 0
        
        # Count validation results
        total_insights = len(self.validations)
        fact_checked_count = sum(1 for v in self.validations if v.get('fact_checked', False))
        bias_detected_count = sum(1 for v in self.validations if v.get('bias_detected', False))
        high_credibility_count = sum(1 for v in self.validations if v.get('credibility_score', 0) > 0.7)
        
        # Identify potential gaps or issues
        gaps_identified = []
        issues_found = []
        
        for validation in self.validations:
            if validation.get('issues_identified') and validation['issues_identified'] not in ['None identified', 'Limited validation (no LLM available)']:
                issues_found.append(validation['issues_identified'])
        
        if fact_checked_count < total_insights * 0.5:
            gaps_identified.append("Insufficient fact verification")
        if calculated_overall_confidence < 0.6:
            gaps_identified.append("Low overall confidence in sources")
        if bias_detected_count > total_insights * 0.3:
            gaps_identified.append("Potential bias detected in multiple sources")
        if high_credibility_count < total_insights * 0.5:
            gaps_identified.append("Limited high-credibility sources")
        
        if not gaps_identified:
            gaps_identified = ["No significant gaps identified"]
        
        # Create summary
        if self.llm_model and total_insights > 0:
            summary = (
                f"Validation completed for {total_insights} insights. "
                f"{fact_checked_count} passed fact-checking, "
                f"{bias_detected_count} showed potential bias, "
                f"{high_credibility_count} from high-credibility sources. "
                f"Overall confidence: {calculated_overall_confidence:.2f}"
            )
        else:
            summary = f"Basic validation completed for {total_insights} insights (confidence: {calculated_overall_confidence:.2f})"

        overall_validation = {
            "type": "overall_validation",
            "summary": summary,
            "gaps_identified": gaps_identified,
            "issues_found": issues_found,
            "overall_confidence": calculated_overall_confidence,
            "validation_stats": {
                "total_insights": total_insights,
                "fact_checked": fact_checked_count,
                "bias_detected": bias_detected_count,
                "high_credibility": high_credibility_count
            }
        }
        self.validations.append(overall_validation)
        # Add each validation individually to the knowledge base
        for validation_item in self.validations:
            await self.knowledge_base.add_data(query_id, "validated_data", validation_item)
        
        self.logger.info(f"CriticAgent: Finished critical assessment for query ID: {query_id}. Generated {len(self.validations)} validations.")
        return True

    async def report_results(self):
        """
        Reports the validation results.
        """
        if self.validations:
            self.logger.info(f"CriticAgent: Reporting {len(self.validations)} validation results.")
            for validation in self.validations:
                self.logger.info(f"  - Validation: {validation.get('summary', validation.get('insight_summary'))}")
            return self.validations
        else:
            self.logger.info("CriticAgent: No validation results found.")
            return []