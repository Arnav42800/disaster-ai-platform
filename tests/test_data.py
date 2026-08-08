from pathlib import Path

from disaster_ai.config import CLASS_NAMES
from disaster_ai.data import build_manifest, parse_event_from_filename


def test_parse_event_from_filename():
    assert (
        parse_event_from_filename("hurricane-michael_00000123_post_disaster.png")
        == "hurricane-michael"
    )


def test_build_manifest_assigns_fixed_splits(tmp_path: Path):
    for label in CLASS_NAMES:
        (tmp_path / label).mkdir()

    (tmp_path / "destroyed" / "socal-fire_00000001_post_disaster.png").write_bytes(b"x")
    (tmp_path / "no_damage" / "hurricane-michael_00000001_post_disaster.png").write_bytes(b"x")
    (tmp_path / "minor_damage" / "santa-rosa-wildfire_00000001_post_disaster.png").write_bytes(b"x")

    df = build_manifest(tmp_path)

    rows = {(row.label, row.event): row.split for row in df.itertuples()}
    assert rows[("destroyed", "socal-fire")] == "train"
    assert rows[("no_damage", "hurricane-michael")] == "val"
    assert rows[("minor_damage", "santa-rosa-wildfire")] == "test"
    assert list(df.columns) == ["image_path", "label", "event", "split"]


def test_build_manifest_supports_deterministic_stratified_splits(tmp_path: Path):
    for label in CLASS_NAMES:
        (tmp_path / label).mkdir()
        for index in range(10):
            filename = f"event-{label}_{index:08d}_post_disaster.png"
            (tmp_path / label / filename).write_bytes(b"x")

    first = build_manifest(tmp_path, split_strategy="stratified", seed=7)
    second = build_manifest(tmp_path, split_strategy="stratified", seed=7)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}
    assert first.groupby(["split", "label"]).size().min() >= 1
