"""
Tests for LLVM IR Inspector and Vectorization Analyzer.
"""

from profila._llvm import analyze_llvm_ir, render_llvm_report


def test_analyze_llvm_ir_vectorization() -> None:
    sample_llvm = """
    define void @matrix_mult() {
    entry:
      %0 = load <4 x double>, <4 x double>* %ptr
      %1 = fadd <4 x double> %0, %0
      store <4 x double> %1, <4 x double>* %ptr
      br label %vector.body
    vector.body:
      ret void
    }
    """
    res = analyze_llvm_ir("matrix_mult", "sig", sample_llvm)

    assert res.simd_vectorized is True
    assert res.simd_instructions >= 2
    assert res.vector_loop_count >= 1

    report = render_llvm_report([res])
    assert "SIMD Vectorization ENABLED" in report
