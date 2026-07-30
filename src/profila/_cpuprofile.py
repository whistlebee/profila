"""
Chrome DevTools CPU Profile (.cpuprofile) generator for Profila.

Converts sampled stack traces into V8/Chrome DevTools compatible CPU Profile JSON format.
"""

from collections import defaultdict
import json
import os
import time
from typing import Any, Dict, List, Tuple

from ._stats import Frame, Stats


def stats_to_cpuprofile(stats: Stats, sampling_interval_us: int = 1000) -> Dict[str, Any]:
    """
    Converts Stats into V8 / Chrome DevTools .cpuprofile JSON structure.
    """
    # Root node (id: 1)
    nodes: List[Dict[str, Any]] = [
        {
            "id": 1,
            "callFrame": {
                "functionName": "(root)",
                "scriptId": "0",
                "url": "",
                "lineNumber": -1,
                "columnNumber": -1,
            },
            "hitCount": 0,
            "children": [],
        }
    ]

    # Map (url, lineNumber, functionName) -> node_id
    frame_to_id: Dict[Tuple[str, int, str], int] = {}
    node_counter = 1

    def get_or_create_node(filename: str, line: int, func_name: str = "") -> int:
        nonlocal node_counter
        key = (filename, line, func_name or f"line:{line}")
        if key in frame_to_id:
            return frame_to_id[key]

        node_counter += 1
        new_id = node_counter

        func_display = func_name if func_name else os.path.basename(filename)
        node = {
            "id": new_id,
            "callFrame": {
                "functionName": func_display,
                "scriptId": "1",
                "url": filename,
                "lineNumber": line,
                "columnNumber": 0,
            },
            "hitCount": 0,
            "children": [],
        }
        nodes.append(node)
        frame_to_id[key] = new_id
        return new_id

    samples: List[int] = []
    time_deltas: List[int] = []

    # Process bad & non-numba samples
    if stats.bad_samples > 0:
        bad_id = get_or_create_node("[bad_sample]", 0, "[bad_sample]")
        nodes[0]["children"].append(bad_id)
        nodes[bad_id - 1]["hitCount"] += stats.bad_samples
        for _ in range(stats.bad_samples):
            samples.append(bad_id)
            time_deltas.append(sampling_interval_us)

    if stats.other_samples > 0:
        other_id = get_or_create_node("[non_numba]", 0, "[non_numba]")
        nodes[0]["children"].append(other_id)
        nodes[other_id - 1]["hitCount"] += stats.other_samples
        for _ in range(stats.other_samples):
            samples.append(other_id)
            time_deltas.append(sampling_interval_us)

    # Process recorded stacks
    for stack, count in stats.stack_counts.items():
        if not stack:
            continue

        parent_id = 1
        leaf_id = 1

        for frame in stack:
            frame_id = get_or_create_node(frame.file, frame.line, getattr(frame, "name", ""))
            parent_node = nodes[parent_id - 1]
            if frame_id not in parent_node["children"]:
                parent_node["children"].append(frame_id)
            parent_id = frame_id
            leaf_id = frame_id

        # Update hit count for leaf node
        nodes[leaf_id - 1]["hitCount"] += count

        for _ in range(count):
            samples.append(leaf_id)
            time_deltas.append(sampling_interval_us)

    now_us = int(time.time() * 1_000_000)
    total_duration_us = len(samples) * sampling_interval_us

    cpuprofile = {
        "nodes": nodes,
        "startTime": now_us - total_duration_us,
        "endTime": now_us,
        "samples": samples,
        "timeDeltas": time_deltas,
    }
    return cpuprofile


def generate_cpuprofile_json(stats: Stats) -> str:
    """
    Returns formatted JSON string of the .cpuprofile format.
    """
    profile = stats_to_cpuprofile(stats)
    return json.dumps(profile, indent=2)
