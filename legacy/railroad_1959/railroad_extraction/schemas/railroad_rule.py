from pydantic import BaseModel
from typing import List, Optional

class RailroadRule(BaseModel):
    rule_id: str
    text: str
    category: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    sub_rules: Optional[List['RailroadRule']] = None
