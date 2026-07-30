"""
Tests for profiling Numba code using Profila and JIT DWARF resolution.
"""

import os
import subprocess
import sys
import numpy as np
import pytest

from profila._stats import Stats, Frame
from profila._dwarf import JITDWARFResolver, LineMapping
from profila._render import render_text
sys.path.insert(0, os.path.abspath("."))
from scripts_for_tests.numba_target import expensive_numba_calc, simple_loop


def test_numba_function_execution() -> None:
    """
    Verify Numba functions execute correctly under debug mode.
    """
    os.environ["NUMBA_DEBUGINFO"] = "1"
    data = np.linspace(1.0, 100.0, 500)
    res = expensive_numba_calc(data)
    assert len(res) == 500
    assert simple_loop(100) == 7425.0


def test_stats_aggregation_for_numba_lines() -> None:
    """
    Verify Stats properly attributes samples to Numba source lines.
    """
    script_path = os.path.abspath("scripts_for_tests/numba_target.py")
    stats = Stats()

    # Simulate 80 samples on line 13 (expensive line) and 20 on line 16 (cheaper line)
    for _ in range(80):
        stats.add_sample([Frame(file=script_path, line=13)])
    for _ in range(20):
        stats.add_sample([Frame(file=script_path, line=16)])

    final_stats = stats.finalize()
    assert final_stats.total_samples == 100
    assert final_stats.numba_samples[script_path][13] == 80.0
    assert final_stats.numba_samples[script_path][16] == 20.0

    rendered = render_text(final_stats)
    assert "80.0% |" in rendered
    assert "20.0% |" in rendered
    assert "expensive_numba_calc" in rendered or "numba_target.py" in rendered


def test_dwarf_resolver_numba_line_mapping() -> None:
    """
    Test JITDWARFResolver resolves simulated Numba code address ranges.
    """
    script_path = os.path.abspath("scripts_for_tests/numba_target.py")
    resolver = JITDWARFResolver()
    
    # Register line mapping for Numba function body
    resolver.mappings.append(
        LineMapping(
            start_address=0x7ff00000,
            end_address=0x7ff00080,
            filename=script_path,
            line=13,
        )
    )
    resolver.mappings.append(
        LineMapping(
            start_address=0x7ff00080,
            end_address=0x7ff000A0,
            filename=script_path,
            line=16,
        )
    )

    frame1 = resolver.resolve_pc(0x7ff00010)
    assert frame1 is not None
    assert frame1.file == script_path
    assert frame1.line == 13

    frame2 = resolver.resolve_pc(0x7ff00090)
    assert frame2 is not None
    assert frame2.file == script_path
    assert frame2.line == 16


def test_cli_annotate_with_numba_script() -> None:
    """
    Verify profiling execution via `profila annotate`.
    """
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "profila",
            "annotate",
            "--",
            "-c",
            "from scripts_for_tests.numba_target import simple_loop; simple_loop(1000)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    assert p.returncode == 0
    assert b"PROFILA EXECUTION BREAKDOWN" in stdout
