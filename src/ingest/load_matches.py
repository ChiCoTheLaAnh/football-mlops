import json
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/statsbomb-open-data/data")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

COMP_ID = 43
SEASON_ID = 3

def main():
    path = RAW / "matches" / str(COMP_ID) / f"{SEASON_ID}.json"
    matches = json.loads(path.read_text())

    rows = []
    for m in matches:
        rows.append({
            "match_id": m["match_id"],
            "home_team": m["home_team"]["home_team_name"],
            "away_team": m["away_team"]["away_team_name"],
            "match_date": m["match_date"],
            "competition_id": COMP_ID,
            "season_id": SEASON_ID
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "matches.parquet", index=False)
    print("Saved:", OUT / "matches.parquet")
    print("Matches:", len(df))

if __name__ == "__main__":
    main()
