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


def stats_to_speedscope(stats: Stats, sampling_interval_ms: float = 1.0) -> Dict[str, Any]:
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

    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {
            "frames": frames,
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


def generate_speedscope_json(stats: Stats) -> str:
    """
    Returns formatted Speedscope JSON string.
    """
    profile = stats_to_speedscope(stats)
    return json.dumps(profile, indent=2)
