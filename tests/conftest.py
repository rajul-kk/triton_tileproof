"""Loaded by pytest before any test module is collected.

``triton.jit`` decides at decoration time, when ``msync.gpu`` is first imported, whether a
kernel runs in the CPU interpreter. Several test modules import ``msync.gpu`` (directly or
through ``msync.cuda_vm``), so the variable has to be set before the first of them is
collected, not in whichever module happens to need it. Set it yourself to override.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")
