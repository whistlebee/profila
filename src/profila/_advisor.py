"""
Numba Optimization Advisor for Profila.

Analyzes profiled Numba functions and generates Profila optimization recommendations.
"""

import ast
import inspect
import os
from typing import Dict, List, Any

from ._stats import Stats


class NumbaOptimizationAdvisor:
    """
    Analyzes Python source files containing Numba code and suggests optimizations.
    """

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        suggestions = []
        if not file_path or not os.path.exists(file_path):
            return suggestions

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    has_njit = False
                    has_fastmath = False
                    has_parallel = False

                    # Check decorators
                    for dec in node.decorator_list:
                        dec_name = ""
                        if isinstance(dec, ast.Name):
                            dec_name = dec.id
                        elif isinstance(dec, ast.Call):
                            if isinstance(dec.func, ast.Name):
                                dec_name = dec.func.id
                            for kw in dec.keywords:
                                if kw.arg == "fastmath" and getattr(kw.value, "value", False):
                                    has_fastmath = True
                                elif kw.arg == "parallel" and getattr(kw.value, "value", False):
                                    has_parallel = True

                        if dec_name in ("njit", "jit"):
                            has_njit = True

                    if not has_njit:
                        continue

                    # Check for array allocations in loops
                    loop_allocations = []
                    for child in ast.walk(node):
                        if isinstance(child, (ast.For, ast.While)):
                            for sub in ast.walk(child):
                                if isinstance(sub, ast.Call):
                                    func_id = ""
                                    if isinstance(sub.func, ast.Name):
                                        func_id = sub.func.id
                                    elif isinstance(sub.func, ast.Attribute):
                                        func_id = sub.func.attr

                                    if func_id in ("zeros", "empty", "ones", "copy", "array"):
                                        lineno = getattr(sub, "lineno", child.lineno)
                                        loop_allocations.append((lineno, func_id))

                    if loop_allocations:
                        lines_str = ", ".join([f"line {line} ({func})" for line, func in loop_allocations[:3]])
                        suggestions.append({
                            "function": func_name,
                            "file": file_path,
                            "type": "MEMORY_ALLOCATION_IN_LOOP",
                            "severity": "HIGH",
                            "message": f"Array allocation inside loop detected at {lines_str}. Pre-allocate buffers outside loop for higher throughput.",
                        })

                    if not has_fastmath:
                        suggestions.append({
                            "function": func_name,
                            "file": file_path,
                            "type": "SIMD_FASTMATH_DISABLED",
                            "severity": "MEDIUM",
                            "message": f"Enable @njit(fastmath=True) to allow LLVM floating-point SIMD vectorization & auto-vectorizer optimizations.",
                        })

                    if not has_parallel:
                        # Check if function has heavy nested loops suitable for prange
                        for child in ast.walk(node):
                            if isinstance(child, ast.For):
                                suggestions.append({
                                    "function": func_name,
                                    "file": file_path,
                                    "type": "PARALLEL_PRANGE_AVAILABLE",
                                    "severity": "INFO",
                                    "message": f"Consider using @njit(parallel=True) and numba.prange for multi-core thread scaling.",
                                })
                                break
        except Exception:
            pass

        return suggestions


def render_advisor_report(suggestions: List[Dict[str, Any]]) -> str:
    """
    Format advisory suggestions into human-readable report.
    """
    if not suggestions:
        return "💡 **Profila Advisor:** No performance warnings detected. Your Numba code is well-optimized!"

    out = ["💡 **PROFILA NUMBA OPTIMIZATION ADVISOR**", "=" * 56]
    for s in suggestions:
        icon = "🔴" if s["severity"] == "HIGH" else "🟡" if s["severity"] == "MEDIUM" else "🔵"
        out.append(f"{icon} **[{s['severity']}] Function `{s['function']}`** (`{os.path.basename(s['file'])}`):")
        out.append(f"   └─ {s['message']}\n")

    return "\n".join(out)
