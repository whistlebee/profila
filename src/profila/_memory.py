"""
Memory Allocation Profiler for Numba and Python code.

Tracks heap and array allocations (in bytes, KB, MB) per line and call stack using tracemalloc.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import os
import tracemalloc
from typing import Dict, List, Optional, Tuple

from ._stats import Frame
from ._func_resolver import resolve_function_name


def format_bytes(size_bytes: float) -> str:
    """
    Format byte count into human-readable string (B, KB, MB, GB).
    """
    if size_bytes <= 0:
        return "     0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    val = float(size_bytes)
    while val >= 1024.0 and unit_idx < len(units) - 1:
        val /= 1024.0
        unit_idx += 1
    return f"{val:>6.1f} {units[unit_idx]}"


@dataclass
class MemoryStats:
    """
    Holds memory allocation stats indexed by file, line, and stack trace.
    """
    # Map file -> line -> total allocated bytes
    file_line_bytes: defaultdict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Map stack -> total allocated bytes
    stack_bytes: Counter[Tuple[Frame, ...]] = field(default_factory=Counter)
    total_allocated_bytes: int = 0

    def add_allocation(self, stack: List[Frame], size_bytes: int) -> None:
        if size_bytes <= 0 or not stack:
            return

        self.total_allocated_bytes += size_bytes
        self.stack_bytes[tuple(stack)] += size_bytes

        for frame in stack:
            if frame.file.endswith(".py"):
                self.file_line_bytes[frame.file][frame.line] += size_bytes
                return


class MemoryProfiler:
    """
    Tracemalloc-based memory allocation profiler.
    """

    def __init__(self) -> None:
        self.stats = MemoryStats()

    def start(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)

    def stop(self) -> MemoryStats:
        if not tracemalloc.is_tracing():
            return self.stats

        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "*tracemalloc*"),
            tracemalloc.Filter(False, "*_memory.py*"),
            tracemalloc.Filter(False, "*profila/__main__.py*"),
        ))

        for trace in snapshot.traces:
            size_bytes = trace.size
            raw_stack: List[Frame] = []
            for frame in trace.traceback:
                filename = os.path.abspath(frame.filename)
                lineno = frame.lineno
                func_name = resolve_function_name(filename, lineno)
                raw_stack.append(Frame(file=filename, line=lineno, name=func_name))

            if raw_stack:
                raw_stack.reverse()  # Root to leaf
                self.stats.add_allocation(raw_stack, size_bytes)

        return self.stats


def render_memory_text(stats: MemoryStats) -> str:
    """
    Render memory allocation stats into human-readable source code annotations.
    """
    from io import StringIO
    from linecache import getline

    out = StringIO()
    total_str = format_bytes(stats.total_allocated_bytes)
    out.write(f"**Total Memory Allocated:** {total_str.strip()} ({stats.total_allocated_bytes:,} bytes)\n")

    for filename, line_bytes in stats.file_line_bytes.items():
        if not line_bytes:
            continue

        min_line = min(line_bytes)
        max_line = max(line_bytes)

        out.write(f"\nFile `{filename}` (lines {min_line} to {max_line}):\n\n```\n")
        for line_number in range(min_line, max_line + 1):
            code = getline(filename, line_number).rstrip()
            allocated = line_bytes.get(line_number, 0)
            usage = format_bytes(allocated) if allocated > 0 else "        "
            out.write(f"{usage} | {code}\n")
        out.write("```\n")

    return out.getvalue()
