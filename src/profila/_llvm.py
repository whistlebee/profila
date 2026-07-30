"""
LLVM IR Inspector & SIMD Vectorization Analyzer for Numba functions.

Inspects generated LLVM IR from Numba JIT functions and checks for SIMD vectorization,
loop unrolling, and memory instructions.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class LLVMAnalysisResult:
    function_name: str
    signature: str
    llvm_ir: str
    simd_vectorized: bool = False
    vector_loop_count: int = 0
    total_llvm_instructions: int = 0
    simd_instructions: int = 0
    memory_allocations: int = 0
    function_calls: int = 0
    insights: List[str] = field(default_factory=list)


def analyze_llvm_ir(function_name: str, signature: str, llvm_ir: str) -> LLVMAnalysisResult:
    """
    Analyze LLVM IR for SIMD vectorization, instructions, and loop structures.
    """
    lines = llvm_ir.splitlines()
    total_instructions = 0
    simd_instructions = 0
    memory_allocations = 0
    function_calls = 0
    vector_loops = 0

    vector_type_pattern = re.compile(r"<\s*\d+\s*x\s*(float|double|i32|i64)\s*>")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("attributes"):
            continue

        # Count LLVM IR instructions
        if "=" in stripped or stripped.startswith(("store ", "br ", "ret ", "call ", "unreachable")):
            total_instructions += 1

        # Check SIMD vector instructions
        if vector_type_pattern.search(stripped):
            simd_instructions += 1

        # Check memory allocations
        if "malloc" in stripped or "numba_alloc" in stripped or "alloca" in stripped:
            memory_allocations += 1

        # Check function calls
        if stripped.startswith("call ") or " call " in stripped:
            function_calls += 1

        # Check vector loop labels
        if "vector.body" in stripped or "vector.ph" in stripped:
            vector_loops += 1

    simd_vectorized = simd_instructions > 0 or vector_loops > 0

    insights = []
    if simd_vectorized:
        insights.append(f"✨ SIMD Vectorization ENABLED: Detected {simd_instructions} vector operations across {max(1, vector_loops)} vector loops.")
    else:
        insights.append("⚠️ SIMD Vectorization NOT DETECTED: Consider enabling @njit(fastmath=True) or contiguous C-array memory layout.")

    if memory_allocations > 0:
        insights.append(f"📦 Memory Allocations: {memory_allocations} memory allocation instructions found in LLVM IR.")

    return LLVMAnalysisResult(
        function_name=function_name,
        signature=signature,
        llvm_ir=llvm_ir,
        simd_vectorized=simd_vectorized,
        vector_loop_count=vector_loops,
        total_llvm_instructions=total_instructions,
        simd_instructions=simd_instructions,
        memory_allocations=memory_allocations,
        function_calls=function_calls,
        insights=insights,
    )


def extract_numba_llvm(target_module: Any, function_filter: str = "") -> List[LLVMAnalysisResult]:
    """
    Extracts LLVM IR from all compiled Numba functions in a target module/dict.
    """
    results: List[LLVMAnalysisResult] = []

    # Search for Numba Dispatcher instances
    dispatchers = []
    if isinstance(target_module, dict):
        for name, obj in target_module.items():
            if hasattr(obj, "inspect_llvm") and hasattr(obj, "signatures"):
                dispatchers.append((name, obj))
    else:
        for name in dir(target_module):
            obj = getattr(target_module, name)
            if hasattr(obj, "inspect_llvm") and hasattr(obj, "signatures"):
                dispatchers.append((name, obj))

    for name, disp in dispatchers:
        if function_filter and function_filter.lower() not in name.lower():
            continue

        try:
            llvm_dict = disp.inspect_llvm()
            for sig, ir_text in llvm_dict.items():
                sig_str = str(sig)
                analysis = analyze_llvm_ir(name, sig_str, ir_text)
                results.append(analysis)
        except Exception:
            pass

    return results


def render_llvm_report(results: List[LLVMAnalysisResult], show_full_ir: bool = False) -> str:
    """
    Render LLVM IR analysis results into human-readable text.
    """
    if not results:
        return "⚠️ No compiled Numba functions with LLVM IR found."

    out = ["⚙️ **PROFILA LLVM IR & VECTORIZATION ANALYSIS**", "=" * 60]

    for res in results:
        status_icon = "🚀" if res.simd_vectorized else "🐢"
        out.append(f"\n{status_icon} **Function `{res.function_name}`** (`{res.signature}`):")
        out.append(f"   • Total LLVM IR Instructions: {res.total_llvm_instructions}")
        out.append(f"   • SIMD Vector Instructions:  {res.simd_instructions}")
        out.append(f"   • Memory Allocations:       {res.memory_allocations}")
        out.append(f"   • Internal Function Calls:   {res.function_calls}")
        for insight in res.insights:
            out.append(f"   └─ {insight}")

        if show_full_ir:
            out.append("\n```llvm")
            out.append(res.llvm_ir[:4000] + ("\n... [LLVM IR truncated]" if len(res.llvm_ir) > 4000 else ""))
            out.append("```\n")

    return "\n".join(out)
