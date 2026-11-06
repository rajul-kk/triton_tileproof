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
