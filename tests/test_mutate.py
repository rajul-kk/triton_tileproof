import ast

from triton_equiv.kernels import KERNELS
from triton_equiv.mutate import OPERATORS, mutants


def test_every_mutant_parses_and_differs_from_reference():
    for name, spec in KERNELS.items():
        ref = ast.unparse(ast.parse(spec.source))
        seen = set()
        for m in mutants(name, spec.source):
            ast.parse(m.source)
            assert ast.unparse(ast.parse(m.source)) != ref, m.id
            assert m.id not in seen
            seen.add(m.id)


def test_operators_rewrite_what_they_claim():
    src = KERNELS["vector_add"].source
    by_op = {op: list(mutants("vector_add", src, [op])) for op in OPERATORS}
    assert "offs <= n" in by_op["mask_off_by_one"][0].source
    assert "tl.store(out_ptr + offs, x + y)" in by_op["missing_store_mask"][0].source
    assert "tl.program_id(1)" in by_op["wrong_pid_axis"][0].source
    assert "other=1.0" in by_op["wrong_other"][0].source
    # first-order: only one site changed per mutant
    assert by_op["wrong_other"][0].source.count("other=1.0") == 1


def test_fp_operators_rewrite_what_they_claim():
    from triton_equiv.mutate import FP_OPERATORS

    def one(kernel, op):
        ms = list(mutants(kernel, KERNELS[kernel].source, [op]))
        assert len(ms) == 1, (kernel, op, len(ms))
        return ms[0].source

    src = one("row_sum_loop", "reverse_reduction_loop")
    assert "for k_rev in range(0, tl.cdiv(n_cols, BLOCK))" in src
    assert "k = tl.cdiv(n_cols, BLOCK) - 1 - k_rev" in src
    assert "tl.reshape(acc, (2, acc.shape[0] // 2))" in one("row_sum_loop", "split_sum")
    assert "dtype=tl.float16" in one("row_sum_loop", "acc_dtype_downgrade")
    assert "c = acc\n" in one("matmul_fp16", "drop_cast")
    assert "acc.to(tl.bfloat16)" in one("matmul_fp16", "change_cast")
    # vector_add has no loop, reduction, accumulator or cast: no fp mutants.
    assert list(mutants("vector_add", KERNELS["vector_add"].source, FP_OPERATORS)) == []


def test_fp_operator_site_counts():
    from triton_equiv.mutate import FP_OPERATORS, count_sites
    counts = {k: [count_sites(s.source, op) for op in FP_OPERATORS] for k, s in KERNELS.items()}
    assert counts == {
        "vector_add": [0, 0, 0, 0, 0],
        "row_sum": [0, 1, 0, 0, 0],
        "matmul": [1, 0, 1, 0, 0],
        "row_sum_loop": [1, 1, 1, 0, 0],
        "matmul_fp16": [1, 0, 1, 1, 1],
        "softmax": [0, 1, 0, 0, 0],
        "layernorm": [0, 2, 0, 2, 2],
        "attention": [0, 0, 0, 0, 0],
    }


def test_mask_operators_for_rowwise_kernels():
    def one(kernel, op, site=0):
        return next(m for m in mutants(kernel, KERNELS[kernel].source, [op]) if m.site == site).source

    assert "other=0.0" in one("softmax", "wrong_other") and "inf" not in one("softmax", "wrong_other")
    assert "diff = x - mean\n" in one("layernorm", "drop_where_mask")
    assert "mean = tl.sum(x, axis=0) / BLOCK" in one("layernorm", "divisor_block_size", 0)
    assert "var = tl.sum(diff * diff, axis=0) / BLOCK" in one("layernorm", "divisor_block_size", 1)
    assert "s = s\n" in one("attention", "drop_where_mask")
