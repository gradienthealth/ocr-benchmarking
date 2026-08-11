"""Phase 13c — the docTR `parseq` reader. 🟢 PHI-free by construction.

docTR is **self-hosted and local**: the PARSeq checkpoint is fetched once from Mindee's
public CDN into `~/.cache/doctr/models` and inference runs in-environment, so no crop — and
therefore no PHI — ever leaves the box, and no BAA is needed (CLAUDE.md §4). This module
never prints, logs or returns anything but the read string.

**Why a separate reader when docTR already has a runner.** `run_doctr.py` runs docTR
end-to-end at its pretrained defaults, whose recognition head is `crnn_vgg16_bn`. PARSeq is a
different recognizer — an autoregressive transformer rather than a CTC-decoded CRNN — and the
reading arm is precisely where a recognition head can be measured without its detector's
errors folded in. Only the recognition predictor is built here: this reader is handed a box
by the ground truth and has no detection stage at all.

IDENTITY: WHY THIS DOES NOT COLLIDE WITH THE docTR RUNNER (D-13.5)
---------------------------------------------------------------------------------
This reader and `DoctrRunner` share `model_name = "doctr"` and the same `doctr.__version__`.
That pair alone is exactly the collision Phase 13a was written for: before `config_id` and
`config_hash` joined `aggregate()`'s identity key, two arms of one library version merged
into one meaningless average. They stay apart here because
  - `config_id` is `"parseq"` vs the runner's `"stock"`/`"tuned"`, and
  - `config()` declares `reco_arch="parseq"`, so the hashes differ,
and on top of that `run_reading` stamps every row of this arm `arm="reader"` and hashes the
crop pipeline into `config_hash` as well. Do not "simplify" `model_name` into
`"doctr-parseq"`: the shared name is what makes a report show two configurations of one
library, which is the comparison Phase 13 is trying to make.

Cost: deliberately **untagged** — no `@priced(...)`, so `harness.cost.estimate_cost` falls
back to the `self_hosted` rule ($0), which is right for a local model.

⚠️ **Latency caveat (D-9.1):** installed as a CPU-only torch build, so latency from this
reader is CPU-only and is NOT representative of GPU-served production numbers.
"""

from __future__ import annotations

import inspect

import doctr
import numpy as np
from doctr.models import recognition_predictor
from doctr.models.recognition.parseq.pytorch import default_cfgs
from PIL import Image

from harness.reading import Reader

RECO_ARCH = "parseq"


def _resolved_default(param: str) -> object:
    """A `recognition_predictor()` default, read from the live signature.

    Sourced from the library rather than hand-typed for the same reason `version` is
    (CLAUDE.md rule #9), and for the same reason `run_doctr._resolved_arch` exists: the
    plan's assumed default and the installed library's actual default have already disagreed
    once in this project (docTR v1.0.1 ships `fast_base`, not the documented `db_resnet50`).
    A knob left at its library default must record the value that really ran, not the value
    someone believed was the default.
    """
    return inspect.signature(recognition_predictor).parameters[param].default


class DoctrParseqReader(Reader):
    """docTR's PARSeq recognition head on the 13b reading path: crop in, string out."""

    model_name = "doctr"  # same library as DoctrRunner ON PURPOSE — see the module docstring
    version_source = "doctr"  # `version` must equal doctr.__version__ (reader test)
    config_id = "parseq"

    def __init__(self) -> None:
        # Read from the installed library, never hand-typed (rule #9). Resolves to the
        # literal string "v1.0.1" — note docTR includes the "v" prefix.
        self.version: str = doctr.__version__
        self.symmetric_pad = _resolved_default("symmetric_pad")
        # The recognizer's fixed input geometry, from docTR's own config for this arch. It is
        # NOT this arm's preprocessing — `reading.crop_for_reading` has already produced the
        # crop and this module does not touch it — but a release that retrains PARSeq at a
        # different input shape changes every string it returns, and that must reach the hash.
        self.input_shape = list(default_cfgs[RECO_ARCH]["input_shape"])
        self._predictor = None

    def config(self) -> dict[str, object]:
        """Every knob that changes what this reader outputs (D-13.5).

        `reco_arch` is what separates this arm from the docTR runner's stock CRNN head.
        `symmetric_pad` and `input_shape` are resolved off the installed library, so a docTR
        release that changes either produces a different hash — which is correct: it is a
        different configuration and its numbers are not comparable to the old ones.

        `pretrained` is not declared, following `run_doctr.config()`: it is `True` on every
        arm that will ever be scored, so it would add a constant to every hash without
        distinguishing anything. Device is absent for the same reason it is absent there — it
        is not a knob that changes the reading, and declaring it would split one arm into a
        CPU identity and a GPU identity.
        """
        return {
            "reco_arch": RECO_ARCH,
            "symmetric_pad": self.symmetric_pad,
            "input_shape": self.input_shape,
        }

    @property
    def predictor(self):
        """The pretrained recognition predictor, built once on first use and reused after.

        Lazy so constructing a `DoctrParseqReader` — to read its `version` for run metadata,
        or to register it in a test list — costs nothing and needs no weights. Cached so the
        `reading.read_image` timer, which wraps only the `read()` call, measures inference
        rather than a one-off model construction and a 95 MB download.

        `recognition_predictor`, not `ocr_predictor`: building the full pipeline would drag in
        a detector this arm must not have. The ground truth supplies the box; a detector here
        would be a second, unmeasured source of error inside a reading number.
        """
        if self._predictor is None:
            self._predictor = recognition_predictor(arch=RECO_ARCH, pretrained=True)
        return self._predictor

    def charset(self) -> frozenset[str]:
        """Characters this reader can possibly emit, read off docTR's config for this arch.

        Checked by a test against `A-Z0-9-`. PARSeq's docTR vocabulary is Latin-first, so
        this passes today; the test exists because the charset is a property of the *release*,
        not of the architecture, and a reader that cannot represent the character class of a
        patient ID would score badly for a reason unrelated to reading quality.
        """
        return frozenset(default_cfgs[RECO_ARCH]["vocab"])

    def read(self, crop: Image.Image) -> str:
        """Read one preprocessed crop. No re-crop, no re-scale, no filtering.

        The crop arrives already padded, height-normalized and RGB (the pinned constants in
        `reading.py`) and is passed straight through as an `(H, W, 3)` uint8 array, which is
        what docTR's predictors consume — the same channel convention `run_doctr` gets from
        `DocumentFile.from_images`, so no reversal is needed here. docTR's own resize to the
        recognizer's input shape happens inside the model and is part of the model, exactly as
        OpenOCR's is; this arm's preprocessing is not touched.

        ⚠️ **Asymmetry worth reporting, not a preprocessing violation.** docTR's
        `RecognitionPredictor` splits any crop wider than 8x its height into overlapping
        sub-crops, reads each, and reassembles the string. At the pinned `CROP_HEIGHT = 48`
        that fires above ~384px of crop width, i.e. on long tokens. It is a fixed default of
        the predictor class, not an argument `recognition_predictor()` exposes, so it is
        pinned by the docTR version rather than being a hidden knob — but SVTRv2 has no
        equivalent stage, so a wide-token difference between the two readers may be this and
        not reading quality. Say so in the write-up rather than averaging it away.

        Returns the model's raw string with no cleanup and no confidence gate: `normalize()`
        is applied later, at scoring time only, and a low-confidence read is still a read —
        discarding one would hide a false redaction, the headline metric (CLAUDE.md §8). An
        empty string means "nothing readable here", which `read_image` scores as an omission.
        """
        results = self.predictor([np.asarray(crop)])
        # docTR returns one (text, confidence) pair per input crop. The confidence is dropped
        # deliberately: `Reader.read` is a string interface, and this arm scores what the
        # model said, never how sure it was.
        return str(results[0][0]) if results else ""
