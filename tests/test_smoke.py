import importlib

import triton_equiv


def test_version():
    assert triton_equiv.__version__ == "0.0.1"


def test_placeholder_modules_import():
    for name in triton_equiv.__all__:
        mod = importlib.import_module(f"triton_equiv.{name}")
        assert mod.__doc__
