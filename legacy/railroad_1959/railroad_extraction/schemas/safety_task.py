from pydantic import BaseModel
from typing import List

class SafetyTask(BaseModel):
    task_id: str
    description: str
    applicable_rules: List[str]
    expected_outcome: str
