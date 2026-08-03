"""Thin pytest fixtures over tests/synthetic.py — no generation logic lives here.

The factory itself is a plain importable module (tests/synthetic.py) so runner
tests can reuse it without pytest; these fixtures are just convenient handles.
All fixtures are in-memory (D-3.2): a test needing a real file path saves the
PIL.Image into its own tmp_path.
"""

from __future__ import annotations

import pytest

from tests import synthetic


@pytest.fixture()
def scene() -> synthetic.Scene:
    """The fixed synthetic scene: 7 images, >=2 strata, >=2 vendors, 1 blank, 2 hard."""
    return synthetic.make_scene(seed=0)


@pytest.fixture()
def allowlist(scene: synthetic.Scene) -> frozenset[str]:
    return scene.allowlist


@pytest.fixture()
def blank_image():
    """(image, gt) for the blank negative control — gt is always []."""
    return synthetic.make_blank_image()


@pytest.fixture()
def synth_image():
    """The image factory itself, for tests that want custom tokens/seed/hard."""
    return synthetic.make_synthetic_image
