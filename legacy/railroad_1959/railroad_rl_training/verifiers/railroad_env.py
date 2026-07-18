import gymnasium as gym
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any
from .railroad_rubric import RailroadRubric

class RailroadEnv(gym.Env):
    def __init__(self, tasks_path: str = None):
        super().__init__()
        
        # Load tasks
        if tasks_path is None:
            # Default to the extracted tasks
            base_dir = Path(__file__).parent.parent.parent
            tasks_path = base_dir / "data" / "railroad_extracted" / "safety_tasks.json"
            
        self.tasks_path = Path(tasks_path)
        if self.tasks_path.exists():
            with open(self.tasks_path, "r", encoding="utf-8") as f:
                self.tasks = json.load(f)
        else:
            print(f"Warning: Tasks file not found at {self.tasks_path}")
            self.tasks = []
            
        self.rubric = RailroadRubric()
        self.current_task = None
        
        # Define spaces (Text-based, so these are placeholders)
        self.observation_space = gym.spaces.Text(max_length=4096)
        self.action_space = gym.spaces.Text(max_length=4096)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        
        if not self.tasks:
            return "No tasks available.", {}
            
        self.current_task = random.choice(self.tasks)
        
        observation = f"""Task ID: {self.current_task['task_id']}
Scenario: {self.current_task['description']}

Instructions: Provide the correct response or action according to the 1959 Consolidated Code of Operating Rules."""
        
        return observation, {"task_id": self.current_task['task_id']}

    def step(self, action: str):
        if not self.current_task:
            return "No active task", 0.0, True, False, {}
            
        # Evaluate response
        scores = self.rubric.evaluate(
            task_description=self.current_task['description'],
            expected_outcome=self.current_task['expected_outcome'],
            agent_response=action
        )
        
        # Calculate composite reward
        # Weights: Safety 0.5, Procedure 0.3, Terminology 0.2
        reward = (
            0.5 * scores.get("safety", 0.0) +
            0.3 * scores.get("procedure", 0.0) +
            0.2 * scores.get("terminology", 0.0)
        )
        
        info = {
            "scores": scores,
            "task_id": self.current_task['task_id']
        }
        
        return "Task Complete", reward, True, False, info
