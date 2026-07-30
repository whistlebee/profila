"""
Tests for Chrome DevTools .cpuprofile generator.
"""

import json
from profila._stats import Stats, Frame
from profila._cpuprofile import stats_to_cpuprofile, generate_cpuprofile_json


def test_cpuprofile_structure() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="/tmp/main.py", line=10), Frame(file="/tmp/math.py", line=25)])
    stats.add_sample([Frame(file="/tmp/main.py", line=10), Frame(file="/tmp/math.py", line=25)])

    profile = stats_to_cpuprofile(stats)

    assert "nodes" in profile
    assert "startTime" in profile
    assert "endTime" in profile
    assert "samples" in profile
    assert "timeDeltas" in profile

    assert len(profile["samples"]) == 2
    assert len(profile["nodes"]) >= 2


def test_generate_cpuprofile_json() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="/tmp/script.py", line=42)])

    raw_json = generate_cpuprofile_json(stats)
    parsed = json.loads(raw_json)

    assert parsed["nodes"][0]["callFrame"]["functionName"] == "(root)"
    assert parsed["samples"] == [2]
