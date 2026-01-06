
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def compute_minutes(row):
    # period mapping: 1(0-45),2(45-90),3(90-105),4(105-120)
    base = {1: 0, 2: 45, 3: 90, 4: 105}.get(int(row["period"]), 0)
    return base + float(row["minute"]) + float(row["second"]) / 60.0

def main():
    events = pd.read_parquet("data/interim/events.parquet")
    matches = pd.read_parquet("data/interim/matches.parquet")

    # Build a home/away mapping
    m = matches.set_index("match_id")[["home_team", "away_team"]].to_dict("index")

    # Keep minimal columns
    keep = ["match_id", "period", "minute", "second", "type", "team", "shot"]
    events = events[keep].dropna(subset=["match_id", "period", "minute", "second", "type"])

    # Add match minute
    events["match_minute"] = events.apply(compute_minutes, axis=1)

    # Goal events: in StatsBomb, goals come from Shot outcome == Goal
    def is_goal(shot_obj):
        if not isinstance(shot_obj, dict):
            return 0
        return 1 if shot_obj.get("outcome", {}).get("name") == "Goal" else 0

    # xG from shot if present (StatsBomb field statsbomb_xg)
    def get_xg(shot_obj):
        if not isinstance(shot_obj, dict):
            return 0.0
        v = shot_obj.get("statsbomb_xg", 0.0)
        return float(v) if v is not None else 0.0

    events["is_goal"] = events["shot"].apply(is_goal).astype(int)
    events["shot_xg"] = events["shot"].apply(get_xg).astype(float)

    # Red card proxy: StatsBomb uses "Foul Committed" + card? but not stored in our flattened schema.
    # For demo, we keep red_card feature = 0.
    events["red_card"] = 0

    rows = []

    for match_id, g in events.groupby("match_id", sort=False):
        if match_id not in m:
            continue
        home = m[match_id]["home_team"]
        away = m[match_id]["away_team"]

        g = g.sort_values("match_minute")

        home_goals = 0
        away_goals = 0
        home_xg = 0.0
        away_xg = 0.0

        # snapshot every 5 minutes (+ at each goal implicitly via running totals)
        snapshot_minutes = set(np.arange(0, 95, 5).tolist())  # 0..90
        next_snap_idx = 0
        snap_list = sorted(snapshot_minutes)

        i_snap = 0

        for _, e in g.iterrows():
            t = float(e["match_minute"])
            team = e["team"]

            # take snapshots up to current time
            while i_snap < len(snap_list) and snap_list[i_snap] <= t:
                tm = snap_list[i_snap]
                rows.append({
                    "match_id": match_id,
                    "home_team": home,
                    "away_team": away,
                    "minute": float(tm),
                    "score_diff": float(home_goals - away_goals),
                    "home_xg": float(home_xg),
                    "away_xg": float(away_xg),
                    "xg_diff": float(home_xg - away_xg),
                    "home_red": 0,
                    "away_red": 0,
                })
                i_snap += 1

            # update running totals
            if e["type"] == "Shot":
                if team == home:
                    home_xg += float(e["shot_xg"])
                    if int(e["is_goal"]) == 1:
                        home_goals += 1
                elif team == away:
                    away_xg += float(e["shot_xg"])
                    if int(e["is_goal"]) == 1:
                        away_goals += 1

        # ensure final snapshots to 90
        while i_snap < len(snap_list):
            tm = snap_list[i_snap]
            rows.append({
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "minute": float(tm),
                "score_diff": float(home_goals - away_goals),
                "home_xg": float(home_xg),
                "away_xg": float(away_xg),
                "xg_diff": float(home_xg - away_xg),
                "home_red": 0,
                "away_red": 0,
            })
            i_snap += 1

    states = pd.DataFrame(rows)

    # Create match outcome labels from final score_diff at 90 snapshot
    final = states.sort_values(["match_id", "minute"]).groupby("match_id").tail(1)[["match_id", "score_diff"]]
    def outcome(sd):
        if sd > 0: return "H"
        if sd < 0: return "A"
        return "D"
    final["outcome"] = final["score_diff"].apply(outcome)

    states = states.merge(final[["match_id", "outcome"]], on="match_id", how="left")

    out_path = OUT / "match_states.parquet"
    states.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print("Rows:", len(states), "Matches:", states["match_id"].nunique())

if __name__ == "__main__":
    main()

