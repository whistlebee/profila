"""
Tests for Flamegraph export and rendering.
"""

from profila._stats import Stats, Frame
from profila._flamegraph import stats_to_flamegraph_tree, generate_flamegraph_html


def test_stats_to_folded_stacks() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="foo.py", line=10), Frame(file="bar.py", line=20)])
    stats.add_sample([Frame(file="foo.py", line=10), Frame(file="bar.py", line=20)])
    stats.add_sample(None)  # bad sample

    folded = stats.to_folded_stacks()
    assert "[bad_sample] 1" in folded
    assert "foo.py:10;bar.py:20 2" in folded


def test_flamegraph_tree() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="/tmp/main.py", line=5), Frame(file="/tmp/helper.py", line=12)])
    
    tree = stats_to_flamegraph_tree(stats)
    assert tree["name"] == "all"
    assert tree["value"] == 1
    assert tree["children"][0]["name"] == "main.py (main.py:5)"
    assert tree["children"][0]["children"][0]["name"] == "helper.py (helper.py:12)"


def test_generate_flamegraph_html() -> None:
    stats = Stats()
    stats.add_sample([Frame(file="test.py", line=100)])
    
    html_content = generate_flamegraph_html(stats, title="Test Flamegraph")
    assert "<!DOCTYPE html>" in html_content
    assert "Test Flamegraph" in html_content
    assert "test.py:100" in html_content
