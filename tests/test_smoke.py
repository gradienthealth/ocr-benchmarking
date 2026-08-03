"""Phase 0 smoke test: the package imports and the test suite runs green.

All later tests run on SYNTHETIC fixtures only (fake images, fake tokens) — never PHI.
"""


def test_package_imports():
    import harness

    assert harness.__doc__ is not None


def test_core_deps_importable():
    # Core deps are installed and importable; no engine deps expected at Phase 0.
    import jiwer  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401
    import pydicom  # noqa: F401
    import rapidfuzz  # noqa: F401
