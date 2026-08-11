"""Phase 13c — the SVTRv2 reader (OpenOCR's recognizer). 🟢 PHI-free by construction.

SVTRv2 is **self-hosted and local**: `openocr-python` downloads one checkpoint on first use
(ModelScope, falling back to HuggingFace, then a manual GitHub release URL) into
`~/.cache/openocr`, and every inference runs in-environment. No crop and therefore no PHI
ever leaves the box, so no BAA is needed (CLAUDE.md §4). This module never prints, logs or
returns anything but the read string, which its caller (`harness.reading.read_image`) keeps
inside the scoring path.

**This is a READER, not an engine.** It gets a crop and returns a string; it has no detector
in this arm and never appears in the end-to-end ranking. OpenOCR's *detector* is separately
ruled out on measured grounds (plan.md PHASE 9, "Ruled out") and nothing here wires it in:
the only OpenOCR task this module ever constructs is `task='rec'`.

WHY `drop_score` IS SET TO 0.0 AND ALSO TESTED
---------------------------------------------------------------------------------
`OpenOCR(drop_score=...)` defaults to **0.5** and drops every read below it. That is a
seed-time filter — a token the model did read, deleted before anyone can see it — and this
project filters at view time, never at read time (CLAUDE.md §8, D-10c.4). A low-confidence
misread of a valid token is a false redaction, the headline metric; discarding it before the
scorer would silently improve the number by hiding the failure.

Two facts, both load-bearing, both true of `openocr-python` 0.1.5:

1. `OpenOCR.__init__` forwards `drop_score` ONLY to the `'ocr'` task — `_init_rec_task`
   never receives it, and `OpenRecognizer` has no such parameter. So the recognition path
   applies no score filter *today*, whatever is passed.
2. This module passes `drop_score=0.0` anyway. It is inert now, but it is the argument that
   would take effect the moment a release wires filtering into the rec path, and the failure
   it guards against is silent: reads would simply stop arriving.

Because (2) is inert, it cannot be trusted on its own — `tests/test_readers.py` asserts (1)
directly against the installed library, so a release that starts filtering fails a test
instead of quietly shrinking the denominator.

NO STRAY OUTPUT FILES
---------------------------------------------------------------------------------
OpenOCR's end-to-end task writes `./e2e_results/system_results.txt` into the current working
directory (`tools/infer_e2e.py`). On real data that file is engine-read token text — PHI —
landing wherever the process happened to be started. Two things keep it from ever appearing:
this module never constructs the `'ocr'` task, and `OpenRecognizer.__call__` writes nothing
(its `./rec_results/` write lives in `infer_rec.main()`, the CLI entry point, which is not on
any path here). A test asserts no file appears in the working directory across a read.

Cost: deliberately **untagged** — no `@priced(...)`, so `harness.cost.estimate_cost` falls
back to the `self_hosted` rule ($0), which is right for a local model.

⚠️ **Latency caveat (D-9.1):** the environment is a CPU-only torch build, so latency from
this reader is CPU-only and is NOT representative of GPU-served production numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openocr
from openocr import OpenOCR
from PIL import Image

from harness.reading import Reader

# The OpenOCR recognition mode this reader is built on. 'server' resolves to the SVTRv2
# checkpoint proper; 'mobile' resolves to RepSVTR, a different (smaller) architecture, which
# is why the mode is part of `config()` and of `config_id` rather than a hidden default.
SERVER_MODE = "server"


def _resolved_rec_config(mode: str) -> dict[str, str]:
    """Architecture and charset OpenOCR selects for `mode`, read from its OWN config files.

    Sourced from the library rather than hand-typed, for the same reason `version` is
    (CLAUDE.md rule #9): if an `openocr-python` release repoints `mode='server'` at a
    different architecture or dictionary, that changes what the reader outputs, and the
    change has to reach `config_hash()` so `aggregate()` refuses to blend the new numbers
    with the old ones. A literal `"SVTRv2_server"` here would go stale in silence.

    Reads YAML only — no weights, no model construction — so `config()` stays cheap and can
    be called (and hashed) before anything is loaded.

    The imports are function-local because `tools` is only importable after `import openocr`
    has run: OpenOCR's package `__init__` puts its own directory on `sys.path` and its
    modules import each other by that top-level name.
    """
    from tools.engine.config import Config
    from tools.infer_rec import (
        DEFAULT_CFG_PATH_REC,
        DEFAULT_CFG_PATH_REC_SERVER,
        DEFAULT_DICT_PATH_REC,
    )

    cfg = Config(DEFAULT_CFG_PATH_REC_SERVER if mode == SERVER_MODE else DEFAULT_CFG_PATH_REC).cfg
    return {
        "algorithm": str(cfg["Architecture"]["algorithm"]),
        # `OpenRecognizer._init_common` overrides the config's own dictionary with this one
        # for every SVTRv2 algorithm, so it — not the YAML's `character_dict_path` — is the
        # charset that actually decodes. Stored by BASENAME: the absolute path is a
        # site-packages location that differs per machine and would make the config hash
        # machine-specific, which would split one arm into two identities for no reason.
        "character_dict": Path(DEFAULT_DICT_PATH_REC).name,
    }


class SVTRv2Reader(Reader):
    """SVTRv2 on the 13b reading path: preprocessed crop in, raw string out."""

    model_name = "svtrv2"
    version_source = "openocr"  # `version` must equal openocr.__version__ (reader test)

    # Inert today (see the module docstring): the rec task never receives it. Declared as an
    # attribute anyway so it is one edit away from being wrong on purpose rather than being
    # a magic literal, and so `config()` records the value this arm ran with.
    drop_score = 0.0

    # OpenOCR's default backend is 'onnx', and `OpenRecognizer` silently switches server mode
    # to torch (the ONNX export exists for the mobile model only). Asking for torch directly
    # means the resolved backend is the one this reader declares rather than one a warning
    # log announced after the fact.
    backend = "torch"

    def __init__(self, *, mode: str = SERVER_MODE) -> None:
        # Read from the installed library, never hand-typed (rule #9). Resolves to "0.1.5".
        self.version: str = openocr.__version__
        self.mode = mode
        # DERIVED from the knob, not passed in — the same rule the docTR runner follows for
        # stock/tuned: a human label that has to be remembered is a label that eventually
        # lies about which config ran.
        self.config_id: str = mode
        self._resolved = _resolved_rec_config(mode)
        self._engine: OpenOCR | None = None

    def config(self) -> dict[str, object]:
        """Every knob that changes what this reader outputs (D-13.5).

        `mode` picks the checkpoint, `algorithm` and `character_dict` are what that mode
        resolves to inside the installed library, `backend` decides which runtime executes
        the graph, and `drop_score` is the filter that must stay at 0.0.

        Device is deliberately absent, following `Runner.config()`'s rule: it is not a knob
        that changes the reading, and declaring it would split one arm into a CPU identity
        and a GPU identity. The latency caveat above is how the CPU build gets reported.
        """
        return {
            "mode": self.mode,
            "backend": self.backend,
            "drop_score": self.drop_score,
            **self._resolved,
        }

    @property
    def engine(self) -> OpenOCR:
        """The recognizer, built once on first use and reused after.

        Lazy so constructing an `SVTRv2Reader` — to read its `version` for run metadata, or
        to register it in a test list — costs nothing and needs no weights. Cached so the
        `reading.read_image` timer, which wraps only the `read()` call, measures inference
        rather than a one-off model construction and a 124 MB download.
        """
        if self._engine is None:
            self._engine = OpenOCR(
                task="rec",  # recognition only — OpenOCR's detector is ruled out
                mode=self.mode,
                backend=self.backend,
                drop_score=self.drop_score,
            )
        return self._engine

    def charset(self) -> frozenset[str]:
        """Characters this reader can possibly emit, read off the loaded decoder.

        Checked by a test against `A-Z0-9-`: OpenOCR's recognizers ship a Chinese-first
        dictionary, and a reader that cannot represent the character class of a patient ID
        would score badly for a reason that has nothing to do with reading quality.
        """
        return frozenset(self.engine.model.post_process_class.character)

    def read(self, crop: Image.Image) -> str:
        """Read one preprocessed crop. No re-crop, no re-scale, no filtering.

        The crop arrives already padded, height-normalized and RGB (the pinned constants in
        `reading.py`); nothing here undoes that — it is what makes two readers' numbers
        comparable. OpenOCR's own resize inside the recognizer is not preprocessing this arm
        controls, it is part of the model, exactly as docTR's is.

        **Channel order.** Passing `img_numpy` skips OpenOCR's first pipeline op, its
        `DecodeImage`, which the eval config declares as `img_mode: BGR`; the array handed
        over must therefore already be BGR. On this project's data the reversal is a no-op —
        renders are grayscale, so R == G == B — but it is done anyway so the reader stays
        correct if it is ever pointed at a color source, and so nobody has to rediscover
        which convention the skipped op assumed.

        Returns the model's raw string with no cleanup and no confidence gate: `normalize()`
        is applied later, at scoring time only, and a low-confidence read is a read (see the
        module docstring). An empty string means "nothing readable here", which `read_image`
        scores as an omission — a real answer, not a failure.
        """
        bgr = np.ascontiguousarray(np.asarray(crop)[:, :, ::-1])
        results = self.engine(img_numpy=bgr)
        return str(results[0]["text"]) if results else ""
