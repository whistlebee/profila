"""
Run Profila as a command-line tool.
"""

from argparse import ArgumentParser, REMAINDER, RawDescriptionHelpFormatter, Namespace
import os
import runpy
import sys

from ._stats import Stats
from ._sampler import Sampler
from ._render import render_text
from ._flamegraph import generate_flamegraph_html
from ._cpuprofile import generate_cpuprofile_json
from ._speedscope import generate_speedscope_json
from ._memory import MemoryProfiler, render_memory_text

PARSER = ArgumentParser(prog="profila", description="A profiler for Numba using direct LLVM DWARF unwinding.")
SUBPARSERS = PARSER.add_subparsers()

ANNOTATE_PARSER = SUBPARSERS.add_parser(
    "annotate",
    help="Run a Python program and annotate the Numba source code.",
    formatter_class=RawDescriptionHelpFormatter,
    description="""To profile "python example.py":

    python -m profila annotate -- example.py

To profile memory allocations:

    python -m profila annotate --mode=memory -- example.py
""",
)
ANNOTATE_PARSER.add_argument(
    "--format",
    choices=["text", "folded", "flamegraph", "cpuprofile", "speedscope"],
    default="text",
    help="Output format: 'text', 'folded', 'flamegraph', 'cpuprofile', or 'speedscope'.",
)
ANNOTATE_PARSER.add_argument(
    "--mode",
    choices=["time", "memory"],
    default="time",
    help="Profiling mode: 'time' (CPU execution sampling) or 'memory' (heap memory allocation tracking).",
)
ANNOTATE_PARSER.add_argument(
    "-o", "--output",
    help="File path to save the output.",
)
ANNOTATE_PARSER.add_argument(
    "--frequency",
    type=float,
    default=1000.0,
    help="Sampling frequency in Hz (default: 1000 Hz / 1ms).",
)
ANNOTATE_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
ANNOTATE_PARSER.set_defaults(command="annotate")

FLAMEGRAPH_PARSER = SUBPARSERS.add_parser(
    "flamegraph",
    help="Profile script and output an interactive flamegraph HTML file.",
)
FLAMEGRAPH_PARSER.add_argument(
    "-o", "--output",
    default="flamegraph.html",
    help="Output HTML file path (default: flamegraph.html).",
)
FLAMEGRAPH_PARSER.add_argument(
    "--frequency",
    type=float,
    default=1000.0,
    help="Sampling frequency in Hz (default: 1000 Hz).",
)
FLAMEGRAPH_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
FLAMEGRAPH_PARSER.set_defaults(command="flamegraph")

CPUPROFILE_PARSER = SUBPARSERS.add_parser(
    "cpuprofile",
    help="Profile script and output a Chrome DevTools .cpuprofile JSON file.",
)
CPUPROFILE_PARSER.add_argument(
    "-o", "--output",
    default="profile.cpuprofile",
    help="Output .cpuprofile file path (default: profile.cpuprofile).",
)
CPUPROFILE_PARSER.add_argument(
    "--frequency",
    type=float,
    default=1000.0,
    help="Sampling frequency in Hz (default: 1000 Hz).",
)
CPUPROFILE_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
CPUPROFILE_PARSER.set_defaults(command="cpuprofile")

SPEEDSCOPE_PARSER = SUBPARSERS.add_parser(
    "speedscope",
    help="Profile script and output a Speedscope JSON (.speedscope.json) file.",
)
SPEEDSCOPE_PARSER.add_argument(
    "-o", "--output",
    default="profile.speedscope.json",
    help="Output Speedscope JSON file path (default: profile.speedscope.json).",
)
SPEEDSCOPE_PARSER.add_argument(
    "--frequency",
    type=float,
    default=1000.0,
    help="Sampling frequency in Hz (default: 1000 Hz).",
)
SPEEDSCOPE_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
SPEEDSCOPE_PARSER.set_defaults(command="speedscope")

MEMORY_PARSER = SUBPARSERS.add_parser(
    "memory",
    help="Profile memory allocations per line of code.",
)
MEMORY_PARSER.add_argument(
    "-o", "--output",
    help="Output file path for annotated memory profile.",
)
MEMORY_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
MEMORY_PARSER.set_defaults(command="memory")

VTUNE_PARSER = SUBPARSERS.add_parser(
    "vtune",
    help="Profile script and launch Profila VTune Edition viewer.",
)
VTUNE_PARSER.add_argument(
    "-o", "--output",
    default="profile.speedscope.json",
    help="Output Speedscope profile path.",
)
VTUNE_PARSER.add_argument(
    "--frequency",
    type=float,
    default=1000.0,
    help="Sampling frequency in Hz (default: 1000 Hz).",
)
VTUNE_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The arguments you'd usually pass to the Python command-line.",
)
VTUNE_PARSER.set_defaults(command="vtune")

ANALYZE_PARSER = SUBPARSERS.add_parser(
    "analyze",
    help="Analyze Numba functions and output VTune-style optimization suggestions.",
)
ANALYZE_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The Python script to analyze.",
)
ANALYZE_PARSER.set_defaults(command="analyze")

LLVM_PARSER = SUBPARSERS.add_parser(
    "llvm",
    help="Inspect LLVM IR and SIMD vectorization status for compiled Numba functions.",
)
LLVM_PARSER.add_argument(
    "--full",
    action="store_true",
    help="Print full LLVM IR assembly text.",
)
LLVM_PARSER.add_argument(
    "--filter",
    default="",
    help="Filter functions by name.",
)
LLVM_PARSER.add_argument(
    "rest",
    nargs=REMAINDER,
    help="The Python script to execute and inspect.",
)
LLVM_PARSER.set_defaults(command="llvm")


def annotate_command(args: Namespace) -> None:
    """
    Run the ``annotate`` command.
    """
    rest = list(args.rest)
    if rest and rest[0] == "--":
        del rest[0]

    if not rest:
        print("Error: No script or program arguments provided.")
        sys.exit(1)

    os.environ["NUMBA_DEBUGINFO"] = "1"

    is_module = False
    is_code = False
    if rest[0] == "-m":
        is_module = True
        del rest[0]
        if not rest:
            print("Error: Missing module name after -m.")
            sys.exit(1)
        target = rest[0]
    elif rest[0] == "-c":
        is_code = True
        del rest[0]
        if not rest:
            print("Error: Missing code string after -c.")
            sys.exit(1)
        target = rest[0]
    else:
        target = rest[0]

    sys.argv = rest

    mode = getattr(args, "mode", "time")
    
    if mode == "memory":
        mem_profiler = MemoryProfiler()
        mem_profiler.start()
        try:
            if is_module:
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            elif is_code:
                exec(target, {"__name__": "__main__"})
            else:
                runpy.run_path(target, run_name="__main__")
        finally:
            mem_stats = mem_profiler.stop()

        output_content = render_memory_text(mem_stats)
    else:
        freq = getattr(args, "frequency", 1000.0)
        interval = 1.0 / max(1.0, freq)

        sampler = Sampler(interval_seconds=interval)
        sampler.start()

        try:
            if is_module:
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            elif is_code:
                exec(target, {"__name__": "__main__"})
            else:
                runpy.run_path(target, run_name="__main__")
        finally:
            stats = sampler.stop()

        out_format = getattr(args, "format", "text")
        if out_format == "folded":
            output_content = stats.to_folded_stacks()
        elif out_format == "flamegraph":
            output_content = generate_flamegraph_html(stats)
        elif out_format == "cpuprofile":
            output_content = generate_cpuprofile_json(stats)
        elif out_format == "speedscope":
            output_content = generate_speedscope_json(stats)
        else:
            final_stats = stats.finalize()
            output_content = render_text(final_stats)

    out_file = args.output
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_content)
        print(f"Profiling results written to {out_file}")
    else:
        if mode == "time" and getattr(args, "format", "text") == "flamegraph":
            with open("flamegraph.html", "w", encoding="utf-8") as f:
                f.write(output_content)
            print("Flamegraph written to flamegraph.html")
        elif mode == "time" and getattr(args, "format", "text") == "cpuprofile":
            with open("profile.cpuprofile", "w", encoding="utf-8") as f:
                f.write(output_content)
            print("Chrome CPU Profile written to profile.cpuprofile")
        elif mode == "time" and getattr(args, "format", "text") == "speedscope":
            with open("profile.speedscope.json", "w", encoding="utf-8") as f:
                f.write(output_content)
            print("Speedscope profile written to profile.speedscope.json")
        else:
            print(output_content)


def flamegraph_command(args: Namespace) -> None:
    args.format = "flamegraph"
    args.mode = "time"
    annotate_command(args)


def cpuprofile_command(args: Namespace) -> None:
    args.format = "cpuprofile"
    args.mode = "time"
    annotate_command(args)


def speedscope_command(args: Namespace) -> None:
    args.format = "speedscope"
    args.mode = "time"
    annotate_command(args)


def memory_command(args: Namespace) -> None:
    args.mode = "memory"
    args.format = "text"
    annotate_command(args)


def vtune_command(args: Namespace) -> None:
    args.format = "speedscope"
    args.mode = "time"
    annotate_command(args)
    vtune_app = os.path.abspath("speedscope-profila/dist/release/index.html")
    if os.path.exists(vtune_app):
        print(f"\n⚡ Profila VTune Edition ready! Open in browser/Helium:\n   file://{vtune_app}")


def analyze_command(args: Namespace) -> None:
    from ._advisor import NumbaOptimizationAdvisor, render_advisor_report

    rest = list(args.rest)
    if rest and rest[0] == "--":
        del rest[0]

    target = rest[0] if rest else ""
    if not target or not os.path.exists(target):
        print("Error: Target Python script file not found.")
        sys.exit(1)

    advisor = NumbaOptimizationAdvisor()
    suggestions = advisor.analyze_file(target)
    print(render_advisor_report(suggestions))


def llvm_command(args: Namespace) -> None:
    from ._llvm import extract_numba_llvm, render_llvm_report

    rest = list(args.rest)
    if rest and rest[0] == "--":
        del rest[0]

    if not rest:
        print("Error: Target Python script not provided.")
        sys.exit(1)

    os.environ["NUMBA_DEBUGINFO"] = "1"
    target = rest[0]
    sys.argv = rest

    res_module = runpy.run_path(target, run_name="__main__")

    results = extract_numba_llvm(res_module, function_filter=getattr(args, "filter", ""))
    print(render_llvm_report(results, show_full_ir=getattr(args, "full", False)))


def main() -> None:
    args = PARSER.parse_args()
    cmd = getattr(args, "command", None)
    if cmd == "annotate":
        annotate_command(args)
    elif cmd == "flamegraph":
        flamegraph_command(args)
    elif cmd == "cpuprofile":
        cpuprofile_command(args)
    elif cmd == "speedscope":
        speedscope_command(args)
    elif cmd == "memory":
        memory_command(args)
    elif cmd == "vtune":
        vtune_command(args)
    elif cmd == "analyze":
        analyze_command(args)
    elif cmd == "llvm":
        llvm_command(args)
    else:
        PARSER.print_help()


if __name__ == "__main__":
    main()
