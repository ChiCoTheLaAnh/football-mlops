from pathlib import Path
RAW = Path("data/raw/statsbomb-open-data")

def main():
    comps = RAW / "data" / "competitions.json"
    if not comps.exists():
        raise SystemExit(f"Missing {comps}. Check clone/submodule path.")
    print("OK:", comps)
    print("Size(bytes):", comps.stat().st_size)
if __name__ == "__main__":
    main()
