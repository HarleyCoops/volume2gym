import os
import json
import logging
import anthropic
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
from dotenv import load_dotenv

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a railroad safety student. "
    "Respond to the scenario based on the 1959 Consolidated Code of Operating Rules. "
    "Ensure your response is safe, follows correct procedure, and uses precise terminology."
)

class RailroadRubric(Rubric):
    """
    LLM-based rubric for evaluating railroad safety tasks.
    Scores on Safety, Procedure, and Terminology.
    """
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        super().__init__()
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._last_ledger: Optional[Dict[str, float]] = None

    def score(
        self,
        completion: Messages,
        answer: str,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward using LLM evaluation.
        """
        # Extract student response
        # verifiers passes a list of messages, we need the last assistant message
        student_response = ""
        for msg in reversed(completion):
            if msg.role == "assistant":
                student_response = msg.content
                break
        
        if not student_response:
            return 0.0

        task_description = (info or {}).get("description", "")
        expected_outcome = answer

        # Call LLM for evaluation
        scores = self._evaluate_with_llm(task_description, expected_outcome, student_response)
        
        # Calculate composite reward
        # Weights: Safety 0.5, Procedure 0.3, Terminology 0.2
        reward = (
            0.5 * scores.get("safety", 0.0) +
            0.3 * scores.get("procedure", 0.0) +
            0.2 * scores.get("terminology", 0.0)
        )
        
        self._last_ledger = {
            "safety": scores.get("safety", 0.0),
            "procedure": scores.get("procedure", 0.0),
            "terminology": scores.get("terminology", 0.0),
            "reasoning": scores.get("reasoning", ""),
            "reward_scalar": reward
        }
        
        return reward

    def _evaluate_with_llm(self, task_description: str, expected_outcome: str, agent_response: str) -> Dict[str, Any]:
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
            "reasoning": "Brief explanation."
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
                
            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {"safety": 0.0, "procedure": 0.0, "terminology": 0.0, "reasoning": "Error"}

    def get_last_ledger(self) -> Optional[Dict[str, float]]:
        return self._last_ledger


class RailroadEnv(SingleTurnEnv):
    def __init__(
        self,
        dataset: Dataset,
        system_prompt: str,
        rubric: RailroadRubric,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )
        self.rubric = rubric

    def get_reward_ledger(self) -> Optional[Dict[str, float]]:
        return self.rubric.get_last_ledger()


def load_environment(
    dataset_path: str | Path,
    system_prompt: Optional[str] = None,
    max_examples: int = -1,
) -> vf.Environment:
    """
    Load the Railroad 1959 environment.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to list of dicts with 'question' and 'answer'
    records = []
    for item in data:
        records.append({
            "question": f"Task ID: {item.get('task_id')}\nScenario: {item.get('description')}",
            "answer": item.get("expected_outcome", ""),
            "info": {
                "task_id": item.get("task_id"),
                "description": item.get("description"),
                "applicable_rules": item.get("applicable_rules")
            }
        })

    if max_examples > 0:
        records = records[:max_examples]

    dataset = Dataset.from_list(records)
    rubric = RailroadRubric()
    
    return RailroadEnv(
        dataset=dataset,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric
    )
