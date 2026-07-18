import os
import anthropic
import json
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
root_env = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(root_env)

class RailroadRubric:
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def evaluate(self, task_description: str, expected_outcome: str, agent_response: str) -> Dict[str, float]:
        """
        Evaluate the agent's response against the task and expected outcome.
        Returns a dictionary of component scores.
        """
        system_prompt = """You are a railroad safety examiner. 
        Evaluate the student's response to the given scenario based on the 1959 Consolidated Code of Operating Rules.
        
        Score the response on three components (0.0 to 1.0):
        1. Safety (critical): Does the response ensure the safety of the train and crew? Any violation is 0.0.
        2. Procedure: Does the response follow the correct sequence of steps?
        3. Terminology: Does the response use the precise railroad terminology required?
        
        Output strictly valid JSON:
        {
            "safety": 0.0-1.0,
            "procedure": 0.0-1.0,
            "terminology": 0.0-1.0,
            "reasoning": "Brief explanation of the scores."
        }
        """
        
        user_content = f"""Scenario: {task_description}
        Expected Outcome: {expected_outcome}
        
        Student Response: {agent_response}
        
        Evaluate this response."""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            
            response_text = message.content[0].text
            
            # Cleanup
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            scores = json.loads(response_text)
            return scores

        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {"safety": 0.0, "procedure": 0.0, "terminology": 0.0, "reasoning": "Error"}
