from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class MatchState:
    home_team: str
    away_team: str
    minute: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_red: int = 0
    away_red: int = 0

    @property
    def score_diff(self) -> float:
        return float(self.home_goals - self.away_goals)

    @property
    def xg_diff(self) -> float:
        return float(self.home_xg - self.away_xg)

    def to_features(self):
        return {
            "minute": float(self.minute),
            "score_diff": float(self.score_diff),
            "xg_diff": float(self.xg_diff),
            "home_red": int(self.home_red),
            "away_red": int(self.away_red),
        }

    def to_dict(self):
        d = asdict(self)
        d.update({"score_diff": self.score_diff, "xg_diff": self.xg_diff})
        return d


STORE: Dict[str, MatchState] = {}
