import json, pathlib


def test_catalog_schema():
    root = pathlib.Path(__file__).parents[1]
    path = root / "experiments" / "catalog" / "nib_suite_45.json"
    data = json.loads(path.read_text())
    assert "experiments" in data and isinstance(data["experiments"], list)
    assert len(data["experiments"]) == 45
    for e in data["experiments"]:
        assert set(["id", "phase", "title"]).issubset(e.keys())
