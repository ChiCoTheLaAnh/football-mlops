import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/interim/events.parquet")

shots = df[df["type"] == "Shot"].copy()
shots["is_goal"] = shots["shot"].apply(lambda s: s.get("outcome", {}).get("name") == "Goal")

shots["body_part"] = shots["shot"].apply(lambda s: s.get("body_part", {}).get("name"))
shots["shot_type"] = shots["shot"].apply(lambda s: s.get("type", {}).get("name"))

# Geometry features
dx = 120 - shots["x"]
dy = shots["y"] - 40
shots["distance_to_goal"] = np.sqrt(dx**2 + dy**2)

left = np.arctan2(44 - shots["y"], dx)
right = np.arctan2(36 - shots["y"], dx)
shots["shot_angle"] = np.abs(left - right)

keep = ["match_id","x","y","body_part","shot_type","distance_to_goal","shot_angle","is_goal"]
shots[keep].to_parquet(OUT / "shots.parquet", index=False)

print("Saved:", OUT / "shots.parquet")
print("Shots:", len(shots))
