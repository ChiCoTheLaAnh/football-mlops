def test_stream_payload_contract():
    init = {"match_id": "demo1", "home_team": "A", "away_team": "B"}
    assert set(init.keys()) == {"match_id", "home_team", "away_team"}

    ev = {"match_id": "demo1", "minute": 10, "type": "shot", "team": "home", "xg": 0.12}
    assert ev["type"] in {"shot", "goal", "red"}
    assert ev["team"] in {"home", "away"}
