# enhanced_llm_agent.py
import os
import json
import asyncio
import aiohttp
import re
from typing import Dict, Any, List
from krutrim_cloud import KrutrimCloud
from dotenv import load_dotenv

load_dotenv()

class EnhancedLLMAgent:
    def __init__(self):
        self.client = KrutrimCloud() # Assuming this is your SDK client
        self.model_name = "Llama-3.3-70B-Instruct"
        self.search_session = None

    async def get_search_session(self):
        if self.search_session is None or self.search_session.closed:
            self.search_session = aiohttp.ClientSession()
        return self.search_session

    async def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo API for market data."""
        try:
            session = await self.get_search_session()
            # The DuckDuckGo API is better for simple queries. For complex benchmarks, it may not return results.
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    # Prioritize the abstract text if available
                    if data.get('AbstractText'):
                        results.append({'title': data.get('Heading'), 'content': data.get('AbstractText')})
                    # Add related topics
                    for topic in data.get('RelatedTopics', [])[:max_results - len(results)]:
                        if topic.get('Text'):
                            results.append({'title': 'Related Topic', 'content': topic.get('Text')})
                    print(f"Web search for '{query}' found {len(results)} results.")
                    return results
                else:
                    print(f"Web search failed with status: {response.status}")
                    return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def _should_search_market_data(self, field_name: str) -> bool:
        """Determine if a web search is needed for a given field."""
        search_keywords = ['reach', 'cpc', 'cpm', 'conversion rate', 'market size', 'churn rate']
        return any(keyword in field_name.lower() for keyword in search_keywords)

    def _build_search_query(self, field_name: str, company_context: Dict) -> str:
        """Build a more effective and general web search query."""
        industry = company_context.get('industry', 'fintech')
        location = company_context.get('location', 'India')
        # Create a more natural language query that's less rigid
        return f"{industry} {field_name} benchmarks {location}"

    def _parse_llm_json_response(self, text: str) -> Dict:
        """Safely parse a JSON object from the LLM's text response."""
        # This regex is more robust for finding a JSON object within a larger string
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                print("Failed to decode JSON from LLM response.")
                return {}
        return {}

    async def get_contextual_suggestions(self, sheet_type: str, field_name: str, current_value: float,
                                         related_fields: Dict[str, Any], company_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual AI suggestions with web search integration and intelligent fallback."""
        try:
            market_data_summary = "No specific market data was searched for this field."
            if self._should_search_market_data(field_name):
                search_query = self._build_search_query(field_name, company_context)
                market_data = await self.search_web(search_query)
                if market_data:
                    market_data_summary = "Web Search Results:\n" + "\n".join([f"- {item['content']}" for item in market_data])
                else:
                    # Explicitly state that the search found nothing.
                    market_data_summary = "No specific market data found from web search."

            prompt = f"""
            You are a financial analyst AI for a startup. Your task is to suggest a realistic value for a specific field in a financial model.

            Company Profile:
            - Industry: {company_context.get('industry', 'N/A')}
            - Stage: {company_context.get('stage', 'N/A')}
            - Location: {company_context.get('location', 'N/A')}

            Analysis Request:
            - Financial Sheet: "{sheet_type}"
            - Field to Analyze: "{field_name}"
            - Current Value: {current_value}

            Contextual Data (related fields from the same quarter):
            {json.dumps(related_fields, indent=2)}

            Market Data Context:
            {market_data_summary}

            Instructions:
            1.  Analyze all the provided information.
            2.  Propose a new, realistic, non-zero value for "{field_name}".
            3.  Provide clear reasoning for your suggestion, referencing the company profile and related data.
            4.  IMPORTANT: If 'Market Data Context' is unavailable or lacks specific numbers, you MUST use your general knowledge of the {company_context.get('industry', 'tech')} sector to estimate a reasonable value. In your reasoning, state that you are using a general industry benchmark.
            5.  Provide a confidence score (low, medium, high).
            6.  Respond ONLY with a single JSON object in the specified format.

            Format:
            {{
              "suggested_value": <float>,
              "reasoning": "<string>",
              "confidence": "<low|medium|high>"
            }}
            """

            # This section is for demonstration. In a real scenario, you would uncomment the client call.
            # The mock logic is enhanced to simulate the new fallback capability.
            await asyncio.sleep(0.5) # Simulate network latency
            
            suggested_value = float(current_value) if current_value > 0 else 1000.0
            reasoning = "Based on general industry trends, a baseline value is suggested."
            
            # Simulate more intelligent fallback for 'reach' fields
            if 'reach' in field_name.lower() and 'spend' in related_fields:
                spend = related_fields['spend']
                if spend > 0:
                    # Simple mock logic: reach is 50x spend. An LLM would have a more nuanced model.
                    suggested_value = spend * 50 
                    reasoning = f"As no specific market data was found, this estimate is based on a general benchmark for a {company_context.get('industry', 'fintech')} company in its {company_context.get('stage', 'growth')} stage. A spend of {spend:,.0f} could realistically generate this level of reach."
            
            mock_response_text = json.dumps({
                "suggested_value": suggested_value,
                "reasoning": reasoning,
                "confidence": "medium"
            })
            
            parsed_response = self._parse_llm_json_response(mock_response_text)
            if not parsed_response:
                raise ValueError("LLM response was not in the expected JSON format.")
                
            return parsed_response

        except Exception as e:
            print(f"Error in get_contextual_suggestions: {e}")
            return {
                "suggested_value": current_value,
                "reasoning": f"An error occurred during AI analysis: {e}",
                "confidence": "low"
            }

    async def comprehensive_stress_test(self, base_projections: Dict[str, Any],
                                        company_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive, pre-defined set of stress tests for investors."""
        scenarios = [
            {'name': 'Market Downturn', 'description': 'A severe economic recession impacts customer spending and acquisition channels for 12 months.'},
            {'name': 'Aggressive Competitor', 'description': 'A new, well-funded competitor enters the market, increasing customer acquisition costs by 50% and churn by 20%.'},
            {'name': 'Regulatory Shift', 'description': 'New data privacy or financial regulations increase compliance costs by 30% and slightly slow down user onboarding.'},
            {'name': 'Funding Winter', 'description': 'The venture capital market tightens, making it impossible to raise the next funding round for 18 months, requiring immediate cost-cutting.'}
        ]
        
        tasks = [self._run_single_stress_scenario(s, base_projections, company_context) for s in scenarios]
        stress_test_results = await asyncio.gather(*tasks)

        summary_prompt = f"""
        You are a venture capital analyst. Based on the following stress test results for a startup, provide a final investment analysis.

        Startup Profile:
        - Industry: {company_context.get('industry', 'N/A')}
        - Stage: {company_context.get('stage', 'N/A')}
        - Current Runway: {company_context.get('current_runway_months', 'N/A')} months

        Stress Test Results:
        {json.dumps(stress_test_results, indent=2)}

        Instructions:
        Provide a concise analysis for a potential investor. Respond ONLY with a single JSON object with the following keys:
        - "investment_risk_score": An integer from 1 (very low risk) to 10 (very high risk).
        - "recommended_funding": A recommended seed/series A funding amount in INR to survive these shocks.
        - "runway_under_stress": The estimated remaining operational runway in months under the most severe scenario.
        - "key_risk_factors": A list of the top 2-3 risk factors identified.
        - "mitigation_requirements": A list of 2-3 critical actions the startup must take to mitigate these risks.
        """
        
        await asyncio.sleep(0.5)
        mock_investor_analysis_text = json.dumps({
            "investment_risk_score": 7,
            "recommended_funding": 50000000,
            "runway_under_stress": "9 months",
            "key_risk_factors": ["Market Downturn", "Aggressive Competitor"],
            "mitigation_requirements": [
                "Diversify customer acquisition channels beyond paid marketing.",
                "Develop a feature moat to increase customer stickiness and reduce churn.",
                "Secure at least 18 months of runway in the next funding round."
            ]
        })

        investor_analysis = self._parse_llm_json_response(mock_investor_analysis_text)
        
        return {
            "stress_test_results": stress_test_results,
            "investor_analysis": investor_analysis
        }

    async def _run_single_stress_scenario(self, scenario: Dict, base_projections: Dict, company_context: Dict) -> Dict:
        """Helper to run a single stress test scenario using the LLM."""
        prompt = f"""
        Analyze the impact of a stress test scenario on a startup's financial projections.

        Scenario Name: {scenario['name']}
        Scenario Description: {scenario['description']}

        Company Profile:
        - Industry: {company_context.get('industry', 'N/A')}
        - Stage: {company_context.get('stage', 'N/A')}
        
        Base Projections (Next Quarter):
        {json.dumps(base_projections, indent=2)}

        Instructions:
        Estimate the percentage impact on key metrics. Respond ONLY with a single JSON object in the format:
        {{
          "revenue_impact_percent": <float, e.g., -25.5>,
          "user_growth_impact_percent": <float>,
          "cost_impact_percent": <float>,
          "risk_level": "<low|medium|high|critical>"
        }}
        """
        await asyncio.sleep(0.3)
        if "Downturn" in scenario['name']:
            impact = {"revenue_impact_percent": -30.0, "user_growth_impact_percent": -40.0, "cost_impact_percent": 5.0, "risk_level": "high"}
        elif "Competitor" in scenario['name']:
            impact = {"revenue_impact_percent": -15.0, "user_growth_impact_percent": -20.0, "cost_impact_percent": 25.0, "risk_level": "high"}
        elif "Regulatory" in scenario['name']:
            impact = {"revenue_impact_percent": -5.0, "user_growth_impact_percent": -10.0, "cost_impact_percent": 30.0, "risk_level": "medium"}
        else: # Funding Winter
            impact = {"revenue_impact_percent": -10.0, "user_growth_impact_percent": -50.0, "cost_impact_percent": -40.0, "risk_level": "critical"}

        result = self._parse_llm_json_response(json.dumps(impact))
        
        return {
            "scenario_name": scenario['name'],
            "description": scenario['description'],
            **result
        }

    async def close_session(self):
        if self.search_session and not self.search_session.closed:
            await self.search_session.close()

# Global instance
enhanced_llm_agent = EnhancedLLMAgent()