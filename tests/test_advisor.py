"""
Tests for Numba Optimization Advisor.
"""

from profila._advisor import NumbaOptimizationAdvisor, render_advisor_report


def test_advisor_detection() -> None:
    advisor = NumbaOptimizationAdvisor()
    suggestions = advisor.analyze_file("scripts_for_tests/large_benchmark.py")

    assert len(suggestions) > 0
    funcs = [s["function"] for s in suggestions]
    assert "matrix_multiply" in funcs
    assert "pairwise_distances" in funcs

    report = render_advisor_report(suggestions)
    assert "PROFILA NUMBA OPTIMIZATION ADVISOR" in report
    assert "fastmath=True" in report
