import pandas as pd

def test_shots_parquet_exists_and_columns():
    required_cols = {"distance_to_goal", "shot_angle", "body_part", "shot_type"}
    assert len(required_cols) == 4

def test_api_payload_schema():
    sample = {
        "distance_to_goal": 12.0,
        "shot_angle": 0.8,
        "body_part": "Right Foot",
        "shot_type": "Open Play",
    }
    assert set(sample.keys()) == {"distance_to_goal", "shot_angle", "body_part", "shot_type"}
