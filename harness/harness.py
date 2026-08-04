"""Phase 6 — the one fixed evaluation loop. 🟢 PHI-free.

`run_harness` is the single, engine-agnostic loop from the design doc: for each image, time
the model call, optionally re-read low-confidence words with a verifier, estimate cost, match
predictions to ground truth, score, and aggregate per-stratum + overall. Every engine — docTR,
PaddleOCR, EasyOCR, a self-hosted VLM, a cloud API — flows through THIS loop unchanged; the
only engine-specific line is `run_func(img)`, and each engine's runner (Phase 8+) is what
turns native output into the frozen `OCROutput` contract. `match`, `score`, and `normalize`
are imported frozen and never touched here, so the comparison across engines stays fair.

Two timing rules the design pins:
- **The timer wraps ONLY `run_func`.** Latency is the model's, not the matcher's/scorer's/
  cost estimator's — those all run after `elapsed` is frozen.
- **Primary and verifier time are reported separately.** `elapsed` is the model call;
  `verifier_elapsed` (None when no verifier ran) is the re-read pass, so the gated setup's
  two costs never merge.

PHI-free: this module only orchestrates. It never loads pixels (the runner and verifier do,
from `ImageRef.path`), never inspects `raw_response`, and surfaces only PHI-free aggregates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

from harness.aggregate import aggregate
from harness.contract import GTToken, OCROutput, OCRWord
from harness.cost import estimate_cost
from harness.matching import match
from harness.metrics import score


@dataclass(frozen=True)
class ImageRef:
    """A pixel-free handle to one input image (Phase 6 decision D-6.1).

    Carries exactly what the harness needs WITHOUT loading pixels: dimensions for cost
    estimation, `path` for the runner/verifier to load the crop, and the manifest metadata
    (`stratum`/`modality`/`vendor`/`frame_idx`) that `run_harness` stamps authoritatively
    onto each score row. That stamp is why blank negative-control frames — which have zero
    GT tokens and therefore `None` stratum from `score()` — still aggregate into their real
    stratum, and it satisfies CLAUDE.md rule #7 (take vendor/stratum from the manifest, never
    re-derive them). `w`/`h` are the pixel dimensions of the exact image fed to the engine.
    """

    id: str
    path: str
    w: int
    h: int
    stratum: str
    modality: str
    vendor: str
    frame_idx: int


# A verifier re-reads one word's crop and returns the corrected string. It receives the
# ImageRef (with `path`, so it can load the crop) and the low-confidence word (with `bbox`);
# it returns ONLY a string. The harness keeps the primary's box and swaps just the text —
# a box-free verifier therefore never loses localization (design section 3.3).
VerifierFunc = Callable[[ImageRef, OCRWord], str]

# The model call: one ImageRef in, one OCROutput out. The single engine-specific line.
RunFunc = Callable[[ImageRef], OCROutput]


def apply_verifier(
    out: OCROutput,
    img: ImageRef,
    verifier_func: VerifierFunc,
    conf_threshold: float,
) -> OCROutput:
    """Re-read ONLY words below `conf_threshold`; the final decision uses the verifier's read.

    - A word with confidence < `conf_threshold` is re-read; the final string is the
      verifier's. High-confidence words are left exactly as the primary emitted them.
    - A word whose confidence is `None` has nothing to gate on, so it is never re-read.
    - **Box-free swap (design 3.3):** the verifier returns only a string; the primary's `bbox`
      and `confidence` are preserved via `dataclasses.replace` — only `.text` changes. So the
      corrected word keeps its localization even when the verifier itself is box-free.

    Returns a new `OCROutput` (frozen dataclasses -> `replace`), preserving `raw_response`,
    `model_name`, and `box_free`.
    """
    new_words = []
    for w in out.words:
        if w.confidence is not None and w.confidence < conf_threshold:
            new_text = verifier_func(img, w)  # verifier loads the crop from img.path
            new_words.append(dataclasses.replace(w, text=new_text))  # box + conf kept
        else:
            new_words.append(w)
    return dataclasses.replace(out, words=new_words)


def run_harness(
    images: list[ImageRef],
    run_func: RunFunc,
    ground_truth: dict[str, list[GTToken]],
    *,
    allowlist: set[str],
    verifier_func: Optional[VerifierFunc] = None,
    verifier_model_name: Optional[str] = None,
    verifier_version: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou: float = 0.5,
) -> dict:
    """Run the fixed loop over `images` and return the aggregated result.

    Per image: time `run_func` (and only `run_func`), optionally re-read sub-threshold words
    with the verifier (timed separately), estimate cost from dimensions, match to
    `ground_truth[img.id]`, score, then aggregate per-stratum + overall.

    The verifier arm is active only when BOTH `verifier_func` and `conf_threshold` are given.
    `stratum`/`modality`/`vendor`/`image_id`/`model_name` are stamped onto each row from the
    authoritative `ImageRef`/`OCROutput` (not the GT-derived values, which are `None` on blank
    controls).

    When the arm is active, `verifier_model_name`/`verifier_version` are REQUIRED and must be
    non-blank — the exact same rule #9 protection already applied to the primary engine
    (`OCROutput.version`, D-8.4), extended to the second model in a gated run. Without this, two
    gated runs whose verifier silently changed between them (a version bump, a swapped model)
    would carry identical `(model_name, version)` and merge in `aggregate()` with nothing to
    flag it — the primary engine's identity says nothing about the verifier's.

    Raises:
        ValueError: `verifier_func` + `conf_threshold` are both given but `verifier_model_name`
            or `verifier_version` is missing/blank.
    """
    verifier_active = verifier_func is not None and conf_threshold is not None
    if verifier_active:
        blank_name = verifier_model_name is None or not verifier_model_name.strip()
        blank_version = verifier_version is None or not verifier_version.strip()
        if blank_name or blank_version:
            raise ValueError(
                "run_harness(): the verifier arm is active (verifier_func + conf_threshold "
                "given) but verifier_model_name/verifier_version is missing or blank. Source "
                "the verifier's exact version the same way a runner sources its own (never "
                "hand-typed) — an unversioned gated run cannot be shown to come from the same "
                "verifier build as its neighbours."
            )

    rows = []
    for img in images:
        t0 = perf_counter()
        out = run_func(img)  # the ONLY model-specific line — and the ONLY thing timed
        elapsed = perf_counter() - t0

        verifier_elapsed: Optional[float] = None
        if verifier_func is not None and conf_threshold is not None:
            tv = perf_counter()
            out = apply_verifier(out, img, verifier_func, conf_threshold)
            verifier_elapsed = perf_counter() - tv  # timed separately, AFTER elapsed is frozen

        cost = estimate_cost(img, run_func)
        matched = match(out, ground_truth[img.id], iou)
        row = score(matched, allowlist, elapsed=elapsed, cost=cost)

        # Authoritative stamp (CLAUDE.md rule #7): overrides score()'s GT-derived values,
        # which are None on blank controls. This is a row-dict write, not a change to score().
        row["stratum"] = img.stratum 
        row["modality"] = img.modality
        row["vendor"] = img.vendor
        row["image_id"] = img.id
        row["model_name"] = out.model_name
        row["version"] = out.version
        row["verifier_elapsed"] = verifier_elapsed
        row["verifier_model_name"] = verifier_model_name if verifier_active else None
        row["verifier_version"] = verifier_version if verifier_active else None
        rows.append(row)

    return aggregate(rows)
