"""
Tests for Speedscope JSON profile generator.
"""

import json
from profila._stats import Stats, Frame
from profila._speedscope import stats_to_speedscope, generate_speedscope_json


def test_speedscope_structure() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="/tmp/main.py", line=10, name="main")])
    stats.add_sample([Frame(file="/tmp/main.py", line=10, name="main")])

    profile = stats_to_speedscope(stats)

    assert "$schema" in profile
    assert "shared" in profile
    assert "profiles" in profile
    assert len(profile["shared"]["frames"]) >= 1
    assert profile["profiles"][0]["type"] == "sampled"


def test_generate_speedscope_json() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="/tmp/script.py", line=42, name="compute")])

    raw_json = generate_speedscope_json(stats)
    parsed = json.loads(raw_json)

    assert parsed["exporter"] == "Profila v0.3.5"
    assert "frames" in parsed["shared"]
