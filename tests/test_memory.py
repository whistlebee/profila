"""
Tests for Memory Allocation Profiling.
"""

from profila._memory import MemoryProfiler, MemoryStats, format_bytes, render_memory_text
from profila._stats import Frame


def test_format_bytes() -> None:
    assert format_bytes(0) == "     0 B"
    assert format_bytes(500) == " 500.0 B"
    assert format_bytes(1024) == "   1.0 KB"
    assert format_bytes(1048576) == "   1.0 MB"


def test_memory_stats_aggregation() -> None:
    stats = MemoryStats()
    stats.add_allocation([Frame(file="/tmp/alloc.py", line=10, name="create_array")], 1024 * 1024)
    stats.add_allocation([Frame(file="/tmp/alloc.py", line=10, name="create_array")], 2 * 1024 * 1024)

    assert stats.total_allocated_bytes == 3 * 1024 * 1024
    assert stats.file_line_bytes["/tmp/alloc.py"][10] == 3 * 1024 * 1024

    rendered = render_memory_text(stats)
    assert "3.0 MB" in rendered
    assert "/tmp/alloc.py" in rendered


def test_memory_profiler_execution() -> None:
    profiler = MemoryProfiler()
    profiler.start()

    # Allocate some array memory
    import numpy as np
    arr = np.ones((500, 500), dtype=np.float64)

    stats = profiler.stop()
    assert stats.total_allocated_bytes > 0
