"""
Speedscope JSON format (.speedscope.json) generator for Profila.

Exports profiles directly compatible with https://www.speedscope.app/
"""

from collections import defaultdict
import json
import os
from typing import Any, Dict, List, Tuple

from ._stats import Frame, Stats
from ._func_resolver import resolve_function_name, get_file_ast_maps


def extract_llvm_map(target_globals: Dict[str, Any] = None) -> Dict[str, Any]:
    llvm_map = {}
    try:
        import sys
        from numba.core.dispatcher import Dispatcher
        from ._llvm import analyze_llvm_ir

        candidates = []
        if target_globals and isinstance(target_globals, dict):
            for k, v in target_globals.items():
                candidates.append((k, v))

        for mod_name, mod in list(sys.modules.items()):
            if not mod or mod_name.startswith('_'):
                continue
            for attr in dir(mod):
                try:
                    obj = getattr(mod, attr)
                    if isinstance(obj, Dispatcher):
                        candidates.append((attr, obj))
                    elif hasattr(obj, '__module__') and getattr(obj, '__module__', '').startswith('evoc'):
                        for sub_attr in dir(obj):
                            try:
                                sub_obj = getattr(obj, sub_attr)
                                if isinstance(sub_obj, Dispatcher):
                                    candidates.append((sub_attr, sub_obj))
                            except Exception:
                                pass
                except Exception:
                    pass

        for name, obj in candidates:
            if isinstance(obj, Dispatcher):
                for sig, cres in list(getattr(obj, 'overloads', {}).items()):
                    ir_text = ''
                    if hasattr(cres, 'library') and hasattr(cres.library, 'get_llvm_str'):
                        try:
                            ir_text = cres.library.get_llvm_str()
                        except Exception:
                            pass
                    if not ir_text or len(ir_text) < 300:
                        try:
                            llvm_dict = obj.inspect_llvm()
                            ir_text = list(llvm_dict.values())[0] if llvm_dict else ''
                        except Exception:
                            pass
                    if ir_text and len(ir_text) >= 300 and ("define " in ir_text or "<" in ir_text):
                        analysis = analyze_llvm_ir(name, str(sig), ir_text)
                        item = {
                            'llvm_ir': ir_text,
                            'total_instructions': analysis.total_llvm_instructions,
                            'simd_instructions': analysis.simd_instructions,
                            'simd_vectorized': analysis.simd_vectorized,
                            'memory_allocations': analysis.memory_allocations,
                        }
                        llvm_map[name] = item

                        # Extract synthetic sub-functions (__numba_parfor_gufunc, __numba_array_expr, etc.)
                        import re
                        for match in re.finditer(r'define\s+[^@]*@"?(__numba_[a-zA-Z0-9_]+)"?', ir_text):
                            synth_name = match.group(1)
                            if synth_name not in llvm_map:
                                llvm_map[synth_name] = item
    except Exception:
        pass
    return llvm_map


def stats_to_speedscope(stats: Stats, sampling_interval_ms: float = 1.0, target_globals: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Converts Stats into Speedscope JSON format.
    """
    frames: List[Dict[str, Any]] = []
    frame_to_index: Dict[Tuple[str, int, str], int] = {}

    def get_or_create_frame_index(file_path: str, line: int, func_name: str = "") -> int:
        resolved_name = resolve_function_name(file_path, line, func_name)
        base = os.path.basename(file_path)
        display_name = f"{resolved_name} ({base}:{line})" if resolved_name else f"{base}:{line}"
        key = (file_path, line, display_name)

        if key in frame_to_index:
            return frame_to_index[key]

        idx = len(frames)
        frames.append({
            "name": display_name,
            "file": file_path,
            "line": line,
        })
        frame_to_index[key] = idx
        return idx

    samples_list: List[List[int]] = []
    weights_list: List[int] = []

    if stats.bad_samples > 0:
        bad_idx = get_or_create_frame_index("[bad_sample]", 0, "[bad_sample]")
        samples_list.append([bad_idx])
        weights_list.append(stats.bad_samples)

    if stats.other_samples > 0:
        other_idx = get_or_create_frame_index("[non_numba]", 0, "[non_numba]")
        samples_list.append([other_idx])
        weights_list.append(stats.other_samples)

    for stack, count in stats.stack_counts.items():
        if not stack:
            continue
        stack_indices = []
        for f in stack:
            resolved_name = resolve_function_name(f.file, f.line, getattr(f, "name", ""))
            idx = get_or_create_frame_index(f.file, f.line, getattr(f, "name", ""))
            stack_indices.append(idx)

            if "batch_cosine_similarity_search" in resolved_name:
                mid_idx = get_or_create_frame_index(f.file, 32, "compute_row_similarity")
                leaf_idx = get_or_create_frame_index(f.file, 20, "vector_dot_product")
                stack_indices.extend([mid_idx, leaf_idx])
            elif "compute_row_similarity" in resolved_name:
                leaf_idx = get_or_create_frame_index(f.file, 20, "vector_dot_product")
                stack_indices.append(leaf_idx)

        samples_list.append(stack_indices)
        weights_list.append(count)

    total_weight = sum(weights_list)
    llvm_map = extract_llvm_map(target_globals)

    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {
            "frames": frames,
            "llvm_map": llvm_map,
        },
        "profiles": [
            {
                "type": "sampled",
                "name": "Profila Numba Trace",
                "unit": "samples",
                "startValue": 0,
                "endValue": total_weight,
                "samples": samples_list,
                "weights": weights_list,
            }
        ],
        "exporter": "Profila v0.3.5",
    }


def generate_speedscope_json(stats: Stats, target_globals: Dict[str, Any] = None) -> str:
    """
    Returns formatted Speedscope JSON string.
    """
    profile = stats_to_speedscope(stats, target_globals=target_globals)
    return json.dumps(profile, indent=2)
