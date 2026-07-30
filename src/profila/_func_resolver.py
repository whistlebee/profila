"""
AST-based Function Name & Call Site Resolver for Profila.

Maps file paths and line numbers to enclosing Python/Numba function names and call targets.
"""

import ast
from functools import lru_cache
import os
from typing import Dict, Tuple


@lru_cache(maxsize=256)
def get_file_ast_maps(file_path: str) -> Tuple[Dict[Tuple[int, int], str], Dict[int, str]]:
    """
    Parses a source file with AST and builds:
    1. Function definition range map: (start_line, end_line) -> function_name
    2. Call site line map: lineno -> called_function_name
    """
    func_map: Dict[Tuple[int, int], str] = {}
    call_map: Dict[int, str] = {}

    if not file_path or not os.path.exists(file_path):
        return func_map, call_map

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                if start > 0 and end >= start:
                    func_map[(start, end)] = node.name

            elif isinstance(node, ast.Call):
                lineno = getattr(node, "lineno", 0)
                if lineno > 0:
                    func_node = node.func
                    name = ""
                    if isinstance(func_node, ast.Name):
                        name = func_node.id
                    elif isinstance(func_node, ast.Attribute):
                        name = func_node.attr

                    if name and name not in ("print", "len", "range", "zeros", "empty", "copy", "randn", "time", "seed"):
                        call_map[lineno] = name
    except Exception:
        pass

    return func_map, call_map


def resolve_function_name(file_path: str, lineno: int, fallback_name: str = "") -> str:
    """
    Find enclosing function name or target call for a given file and line number.
    """
    func_map, call_map = get_file_ast_maps(file_path)

    # 1. Check if line has a specific function call site (takes precedence over generic wrappers)
    if lineno in call_map:
        return call_map[lineno]

    # 2. Return valid non-generic fallback name
    if fallback_name and fallback_name not in ("<module>", "<string>", "run_path", "run_module", "main"):
        return fallback_name

    # 3. Check enclosing function definition range
    for (start, end), func_name in func_map.items():
        if start <= lineno <= end and func_name != "main":
            return func_name

    for (start, end), func_name in func_map.items():
        if start <= lineno <= end:
            return func_name

    return fallback_name or os.path.basename(file_path)
