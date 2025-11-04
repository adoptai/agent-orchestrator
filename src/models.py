from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Action:
    id: str
    title: str
    description: str
    endpoint: str
    method: str
    required_inputs: List[str]
    
    def to_embedding_text(self) -> str:
        return f"{self.title}. {self.description}"

@dataclass
class SearchResult:
    action_id: str
    action: Action
    distance: float
    search_method: str
    time_taken: float

@dataclass
class MatchResult:
    match_type: str  # "none", "single", or "multiple"
    actions: List[Action]
    reasoning: str
    search_results: Optional[SearchResult] = None