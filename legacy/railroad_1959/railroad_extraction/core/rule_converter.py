import os
import json
import anthropic
from typing import List
from dotenv import load_dotenv
from pathlib import Path
from ..schemas.railroad_rule import RailroadRule
from ..schemas.safety_task import SafetyTask

# Load environment variables
root_env = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(root_env)

class RuleConverter:
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def convert_to_tasks(self, rule: RailroadRule) -> List[SafetyTask]:
        """
        Convert a railroad rule into a set of safety tasks (scenarios).
        """
        print(f"Generating tasks for Rule {rule.rule_id}...")
        
        system_prompt = """You are a railroad safety instructor. 
        Your goal is to create training scenarios (tasks) based on specific operating rules.
        
        For the provided rule, generate 3-5 diverse scenarios:
        1. Standard operation: A situation where the rule is followed normally.
        2. Edge case: A complex situation where the rule's application is subtle.
        3. Violation check: A scenario where a violation might easily occur if not careful.
        
        Output strictly valid JSON in the following format:
        {
            "tasks": [
                {
                    "task_id": "RuleID-001",
                    "description": "Detailed scenario description describing the train's situation, location, and orders.",
                    "applicable_rules": ["RuleID"],
                    "expected_outcome": "The exact action the engineer/conductor must take to comply."
                }
            ]
        }
        """
        
        user_content = f"""Rule ID: {rule.rule_id}
        Category: {rule.category}
        Text: {rule.text}
        
        Generate training tasks for this rule."""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            
            response_text = message.content[0].text
            
            # Basic cleanup
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response_text)
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(SafetyTask(**t))
                
            return tasks

        except Exception as e:
            print(f"Error generating tasks for rule {rule.rule_id}: {e}")
            return []

if __name__ == "__main__":
    # Test run
    converter = RuleConverter()
    sample_rule = RailroadRule(
        rule_id="99",
        text="When a train stops under circumstances in which it may be overtaken by another train, the flagman must go back immediately with flagman's signals a sufficient distance to insure full protection, placing two torpedoes, and when necessary, in addition, displaying lighted fusees.",
        category="Protection of Trains"
    )
    
    tasks = converter.convert_to_tasks(sample_rule)
    print(f"Generated {len(tasks)} tasks:")
    for t in tasks:
        print(f"- {t.task_id}: {t.description[:50]}...")
