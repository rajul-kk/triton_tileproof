import subprocess, sys, json
KGYM_COMMIT = "3a84417f8c0efaadb215ef638b37d12e71ed20f3"
subprocess.run(["git", "clone", "-q", "https://github.com/hkust-nlp/KernelGYM.git", "/kaggle/working/KernelGYM"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/KernelGYM", "checkout", "-q", KGYM_COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic", "pydantic-settings==2.10.1", "structlog"], check=True)
sys.path.insert(0, "/kaggle/working/KernelGYM")
import torch, triton
from kernelgym.backend import get_backend
from kernelgym.toolkit.kernel_simple.toolkit import KernelSimpleToolkit

readme = open("/kaggle/working/KernelGYM/README.md").read()
# The kernel_code string of README "Example 1: Kernel Simple Task", verbatim.
TQ = "'" * 3
marker = '"kernel_code": ' + TQ
start = readme.index(marker) + len(marker)
KCODE = readme[start:readme.index(TQ, start)]
assert "@triton.jit" in KCODE and "def get_cases" in KCODE
body, cases_part = KCODE.split("def get_cases():")
CASES_NO_TOL = "import torch\ndef get_cases():" + cases_part
CASES_TOL = CASES_NO_TOL.replace('"outputs": expected}', '"outputs": expected, "rtol": 1e-4, "atol": 1e-5}')
assert CASES_TOL != CASES_NO_TOL

backend, tk = get_backend("kernelbench"), KernelSimpleToolkit()


def run(name, **kw):
    task = {"task_id": name, "entry_point": "ModelNew", "backend": "triton", "device": "cuda:0",
            "run_correctness": True, "run_performance": False, **kw}
    r = tk.evaluate(task, backend=backend)
    out = {"case": name, "compiled": r.get("compiled"), "correctness": r.get("correctness"),
           "error": (r.get("error_message") or "")[:300]}
    print(out)
    return out


res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
       "kernelgym": KGYM_COMMIT, "runs": [
           run("A_readme_verbatim", kernel_code=KCODE),
           run("B_cases_code_no_tolerance", kernel_code=body, cases_code=CASES_NO_TOL),
           run("C_cases_code_with_tolerance", kernel_code=body, cases_code=CASES_TOL),
       ]}
json.dump(res, open("/kaggle/working/readme_check.json", "w"), indent=1)
