import json
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/statsbomb-open-data/data")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

matches = pd.read_parquet("data/interim/matches.parquet")
rows = []

for mid in matches["match_id"]:
    path = RAW / "events" / f"{mid}.json"
    events = json.loads(path.read_text())

    for e in events:
        row = {
            "match_id": mid,
            "event_id": e.get("id"),
            "period": e.get("period"),
            "minute": e.get("minute"),
            "second": e.get("second"),
            "type": e["type"]["name"],
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "x": e.get("location", [None, None])[0],
            "y": e.get("location", [None, None])[1],
            "shot": e.get("shot")
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.to_parquet(OUT / "events.parquet", index=False)

print("Saved:", OUT / "events.parquet")
print("Events:", len(df))
