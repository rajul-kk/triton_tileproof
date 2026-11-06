"""Empirical check of one oracle label with Triton's CPU interpreter (Linux/WSL only).

The oracle labels ``vector_add:missing_store_mask`` as OOB at n=6, BLOCK=4 (store to
address 6). We run the reference and that mutant under ``TRITON_INTERPRET=1`` with the
output as a view into a larger buffer whose tail is a canary. The OOB store then lands
in the canary instead of real out-of-bounds memory.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
    pytest.skip("triton/torch not installed (no official Windows wheel; run under WSL/Linux)",
                allow_module_level=True)

SCRIPT = textwrap.dedent('''
    import json, sys, torch
    from triton_equiv.kernels import KERNELS, to_module
    from triton_equiv.mutate import mutants
    spec = KERNELS["vector_add"]
    mut = next(mutants("vector_add", spec.source, ["missing_store_mask"]))
    n, BLOCK, CANARY = 6, 4, -12345.0
    out = {}
    for tag, src in [("ref", spec.source), ("mut", mut.source)]:
        fn = to_module(src, spec.fn_name)
        x, y = torch.arange(n, dtype=torch.float32), torch.ones(n)
        buf = torch.full((n + BLOCK,), CANARY)
        fn[(-(-n // BLOCK),)](x, y, buf[:n], n, BLOCK=BLOCK)
        out[tag] = {"correct": bool(torch.equal(buf[:n], x + y)),
                    "canary_hit": [i for i in range(n, n + BLOCK) if buf[i].item() != CANARY]}
    print(json.dumps(out))
''')


def test_interpreter_confirms_missing_store_mask_oob():
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    res = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["ref"] == {"correct": True, "canary_hit": []}
    # Mutant output is still correct in-bounds (allclose would accept it) ...
    assert out["mut"]["correct"] is True
    # ... but it writes past the end, starting at address 6, matching the oracle witness.
    assert out["mut"]["canary_hit"] and out["mut"]["canary_hit"][0] == 6
