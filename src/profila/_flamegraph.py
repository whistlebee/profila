"""
Flamegraph generator for Profila.

Generates both folded stack trace formats and standalone interactive HTML/SVG flamegraphs.
"""

from collections import defaultdict
import html
import json
import os
from typing import Any, Dict, List
from ._stats import Stats
from ._func_resolver import resolve_function_name

NUMBA_INTERNAL_MODULES = {
    "dispatcher.py",
    "compiler.py",
    "compiler_machinery.py",
    "compiler_lock.py",
    "typed_passes.py",
    "typeinfer.py",
    "lowering.py",
    "codegen.py",
    "ffi.py",
    "executionengine.py",
    "serialize.py",
    "debuginfo.py",
}


def clean_frame_name(file_path: str, line: int, func_name: str = "") -> str:
    base = os.path.basename(file_path)
    if base in NUMBA_INTERNAL_MODULES:
        return "[Numba JIT Internal]"

    resolved = resolve_function_name(file_path, line, func_name)
    if resolved and resolved not in ("<module>", "<string>", "run_path", "run_module"):
        return f"{resolved} ({base}:{line})"
    return f"{base}:{line}"


def process_stack_frames(stack: List[Any]) -> List[str]:
    """
    Format stack frames, collapsing contiguous Numba internal compilation steps.
    """
    cleaned: List[str] = []
    prev_is_internal = False

    for f in stack:
        name = clean_frame_name(f.file, f.line, getattr(f, "name", ""))
        if name == "[Numba JIT Internal]":
            if not prev_is_internal:
                cleaned.append("[Numba JIT Overhead/Compilation]")
                prev_is_internal = True
        else:
            cleaned.append(name)
            prev_is_internal = False

    return cleaned if cleaned else ["[Numba Execution]"]


def stats_to_flamegraph_tree(stats: Stats) -> Dict[str, Any]:
    """
    Convert Stats into a hierarchical tree format suitable for flamegraph visualization.
    Root -> Children -> Leaf Nodes
    """
    tree: Dict[str, Any] = {"name": "all", "value": 0, "children": []}

    def get_or_create_child(parent_node: Dict[str, Any], name: str) -> Dict[str, Any]:
        if "children" not in parent_node:
            parent_node["children"] = []
        for child in parent_node["children"]:
            if child["name"] == name:
                return child
        new_child = {"name": name, "value": 0, "children": []}
        parent_node["children"].append(new_child)
        return new_child

    def add_stack_to_tree(stack_frames: List[str], count: int) -> None:
        curr = tree
        curr["value"] += count
        for frame in stack_frames:
            curr = get_or_create_child(curr, frame)
            curr["value"] += count

    if stats.bad_samples > 0:
        add_stack_to_tree(["[bad_sample]"], stats.bad_samples)
    if stats.other_samples > 0:
        add_stack_to_tree(["[non_numba]"], stats.other_samples)

    for stack, count in stats.stack_counts.items():
        frames = process_stack_frames(list(stack))
        add_stack_to_tree(frames, count)

    return tree


def get_max_depth(node: Dict[str, Any], current_depth: int = 1) -> int:
    if "children" not in node or not node["children"]:
        return current_depth
    return max(get_max_depth(child, current_depth + 1) for child in node["children"])


def generate_flamegraph_html(stats: Stats, title: str = "Profila Numba Flamegraph") -> str:
    """
    Generates a standalone, self-contained interactive HTML Flamegraph.
    """
    tree_data = stats_to_flamegraph_tree(stats)
    json_data = json.dumps(tree_data)
    max_depth = get_max_depth(tree_data)
    svg_height = max(400, max_depth * 26 + 40)

    template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #0f172a; color: #f8fafc; }}
        h2 {{ color: #38bdf8; margin-bottom: 5px; }}
        .info {{ color: #94a3b8; font-size: 14px; margin-bottom: 15px; }}
        #chart {{ width: 100%; min-height: 400px; background: #1e293b; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); box-sizing: border-box; }}
        .node {{ cursor: pointer; stroke: #0f172a; stroke-width: 0.5px; transition: opacity 0.15s; }}
        .node:hover {{ opacity: 0.85; filter: brightness(1.2); }}
        text {{ font-size: 11px; fill: #ffffff; pointer-events: none; text-shadow: 0 1px 2px rgba(0,0,0,0.8); font-weight: 500; }}
        #tooltip {{ position: absolute; display: none; background: rgba(15, 23, 42, 0.95); color: #fff; padding: 8px 12px; border-radius: 6px; border: 1px solid #334155; font-size: 12px; pointer-events: none; z-index: 100; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }}
    </style>
</head>
<body>
    <h2>🔥 {html.escape(title)}</h2>
    <div class="info">Total Samples: {stats.total_samples()} | High-Frequency 1000Hz Profiler</div>
    <div id="chart"></div>
    <div id="tooltip"></div>

    <script>
        const data = {json_data};
        const container = document.getElementById('chart');
        const tooltip = document.getElementById('tooltip');
        
        function renderTree(node, depth, x, width, totalVal) {{
            if (width < 0.002) return '';
            const pct = ((node.value / totalVal) * 100).toFixed(1);
            let hue = (depth * 45 + node.name.length * 17) % 360;
            if (node.name.includes('[Numba JIT')) hue = 30; // Warm Amber for Numba Overhead
            if (node.name.includes('all')) hue = 210;
            
            const color = `hsl(${{hue}}, 65%, 45%)`;
            const y = depth * 24;
            
            let htmlStr = `<g class="node-group" data-name="${{node.name.replace(/"/g, '&quot;')}}" data-val="${{node.value}}" data-pct="${{pct}}">
                <rect class="node" x="${{x * 100}}%" y="${{y}}" width="${{width * 100}}%" height="22" rx="3" fill="${{color}}"></rect>
                ${{width > 0.025 ? `<text x="${{x * 100 + 0.4}}%" y="${{y + 15}}">${{node.name}} (${{pct}}%)</text>` : ''}}
            </g>`;
            
            if (node.children && node.children.length > 0) {{
                let currX = x;
                for (const child of node.children) {{
                    const childWidth = (child.value / totalVal);
                    htmlStr += renderTree(child, depth + 1, currX, childWidth, totalVal);
                    currX += childWidth;
                }}
            }}
            return htmlStr;
        }}

        const total = data.value || 1;
        const svgContent = `<svg width="100%" height="{svg_height}" style="overflow: visible;">${{renderTree(data, 0, 0, 1, total)}}</svg>`;
        container.innerHTML = svgContent;

        document.querySelectorAll('.node-group').forEach(el => {{
            el.addEventListener('mousemove', (e) => {{
                tooltip.style.display = 'block';
                tooltip.style.left = (e.pageX + 12) + 'px';
                tooltip.style.top = (e.pageY + 12) + 'px';
                tooltip.innerHTML = `<strong>${{el.dataset.name}}</strong><br/>Samples: ${{el.dataset.val}} (${{el.dataset.pct}}%)`;
            }});
            el.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
        }});
    </script>
</body>
</html>"""
    return template
