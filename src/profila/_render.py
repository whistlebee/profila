"""
Render ``FinalStats`` to human-readable annotated text, displaying Numba JIT Compilation vs Execution Time Split.
"""

from io import StringIO
from linecache import getline
import os

from ._stats import FinalStats

NUMBA_INTERNAL_PATHS = (
    "site-packages/numba/",
    "numba/core/",
    "numba/np/",
    "numba/misc/",
    "numba/cpython/",
    "lib/python3.",
    "threading.py",
    "profila/",
)


def is_user_file(filename: str) -> bool:
    """
    Returns True if file is user code, False if Numba internal or standard library.
    """
    fn = filename.replace("\\", "/")
    return not any(internal in fn for internal in NUMBA_INTERNAL_PATHS)


def render_text(stats: FinalStats) -> str:
    """
    Render stats to text with clear distinction between User Code and Numba Framework.
    """
    result = StringIO()
    
    user_samples_pct = 0.0
    numba_internal_pct = 0.0

    user_files = {}
    internal_files = {}

    for filename, line_percents in stats.numba_samples.items():
        total_file_pct = sum(line_percents.values())
        if is_user_file(filename):
            user_files[filename] = line_percents
            user_samples_pct += total_file_pct
        else:
            internal_files[filename] = line_percents
            numba_internal_pct += total_file_pct

    result.write("📊 **PROFILA EXECUTION BREAKDOWN**\n")
    result.write(f"• 🎯 **JIT Steady-State Execution:** {stats.percent_execution:.1f}%\n")
    result.write(f"• ⚙️ **JIT Compilation & Lowering:** {stats.percent_compilation:.1f}%\n")
    result.write(f"• 🌐 **Non-Numba / External:** {stats.percent_other_samples:.1f}%\n")
    if stats.percent_bad_samples > 0:
        result.write(f"• ⚠️ **Bad / Unparsed Samples:** {stats.percent_bad_samples:.1f}%\n")
    result.write(f"• ⏱️ **Total Samples Gathered:** {stats.total_samples:,}\n\n")

    # Render User Code Section First
    if user_files:
        result.write("========================================================\n")
        result.write("🎯 **USER CODE ANNOTATIONS**\n")
        result.write("========================================================\n")
        for filename, line_percents in user_files.items():
            if not line_percents:
                continue
            min_line = min(line_percents)
            max_line = max(line_percents)

            result.write(f"\nFile `{filename}` (lines {min_line} to {max_line}):\n\n```\n")
            for line_number in range(min_line, max_line + 1):
                code = getline(filename, line_number).rstrip()
                percent = line_percents.get(line_number, 0)
                usage = f"{percent:>5.1f}%" if percent > 0 else "      "
                result.write(f"{usage} | {code}\n")
            result.write("```\n")

    # Render Numba Internal Framework Section
    if internal_files:
        result.write("\n========================================================\n")
        result.write("⚙️ **NUMBA FRAMEWORK & JIT COMPILATION INTERNALS**\n")
        result.write("========================================================\n")
        for filename, line_percents in internal_files.items():
            if not line_percents:
                continue
            min_line = min(line_percents)
            max_line = max(line_percents)

            result.write(f"\nFile `{filename}` (lines {min_line} to {max_line}):\n\n```\n")
            for line_number in range(min_line, max_line + 1):
                code = getline(filename, line_number).rstrip()
                percent = line_percents.get(line_number, 0)
                usage = f"{percent:>5.1f}%" if percent > 0 else "      "
                result.write(f"{usage} | {code}\n")
            result.write("```\n")

    return result.getvalue()
