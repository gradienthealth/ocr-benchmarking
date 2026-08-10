#!/usr/bin/env python3
"""End-to-end demo of the OCR benchmark harness on 100% SYNTHETIC data. 🟢 PHI-free.

Purpose (a walkthrough you can run live): show what Phases 8 and 9 actually produced, in
three sections that build on each other.

  1. PHASE 8 — the Runner contract + version provenance.
     Every engine is a `Runner` subclass whose `run()` converts native output into the frozen
     `OCROutput`. Each carries a `version` read live from its installed library, and
     `aggregate()` REFUSES to blend rows whose (model_name, version) differ.

  2. PHASE 9 — three REAL local engines through the ONE fixed loop.
     docTR, PP-OCRv6_medium, and EasyOCR each read real synthetic PNGs and are scored by the
     identical matcher/scorer. This is the bake-off, in miniature.

  3. The verifier arm + the cost axis (stub engines).
     All three local engines are $0 and box-carrying, so stubs remain the only way to show a
     priced cloud engine and a verifier recovering a low-confidence misread.

        OCROutput (the frozen contract)                    [contract.py]
          -> match predictions to ground truth by IoU      [matching.py]
          -> score one image (Found/Added, false redaction)[metrics.py]
          -> aggregate per-stratum + overall + neg-control [aggregate.py]
          -> render a stratified markdown report + charts  [report.py]

EVERYTHING here is fake: synthetic images drawn from `tests/synthetic.py` with fake tokens in
the CLAUDE.md house style (CMFN-…/ACC-…/GRDN-…), plus stub engines. No DICOM, no gt.csv, no
GCS, no network. Nothing real-PHI is touched, so it is safe to run and to show.

Run it:  python demo/run_demo.py           (loads real models; see the timing line it prints)
         python demo/run_demo.py --fast    (skips section 2 — no model loading, ~1s)
Output:  markdown reports + charts under demo/output/, and a console summary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Framework deprecation chatter from torch/paddle would otherwise bury the demo's own output.
warnings.filterwarnings("ignore")

# Make `harness` importable when run as `python demo/run_demo.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from harness.aggregate import aggregate  # noqa: E402
from harness.contract import GTToken, OCROutput, OCRWord  # noqa: E402
from harness.cost import priced  # noqa: E402
from harness.harness import ImageRef, run_harness  # noqa: E402
from harness.report import write_report  # noqa: E402

OUT_DIR = _REPO_ROOT / "demo" / "output"
IMG_DIR = OUT_DIR / "images"

RULE = "=" * 78


def _section(title: str) -> None:
    print()
    print(RULE)
    print(title)
    print(RULE)


def _fmt_rate(x):
    return "—" if x is None else f"{x * 100:5.1f}%"


@contextlib.contextmanager
def _quiet():
    """Swallow an engine's own console chatter (load banners, framework notices).

    Redirects at the FILE-DESCRIPTOR level, not via `contextlib.redirect_stdout`: PaddleOCR's
    logger holds a reference to the original stream captured at import time, so swapping
    `sys.stdout` alone does not silence it.

    Discarded rather than forwarded on purpose: on real data an engine's stdout can contain
    read-back token text, so a script that echoes it would leak PHI (CLAUDE.md §7). The buffer
    is surfaced only on an exception — safe here because this demo is synthetic end to end —
    so a genuine failure is never hidden behind the silence.
    """
    saved_out, saved_err = os.dup(1), os.dup(2)
    sink = tempfile.TemporaryFile(mode="w+")
    failed = False
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield
    except BaseException:
        failed = True
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        if failed:
            sink.seek(0)
            print("\n  --- captured engine output (error path; synthetic data only) ---")
            print(sink.read())
        sink.close()


# ===========================================================================
# SECTION 1 — PHASE 8: the Runner contract and version provenance
# ===========================================================================

# Each engine lives behind its own optional extra. A missing one is SHOWN, never hidden —
# "not installed" must never read the same as "fine".
_REAL_RUNNERS: list = []
_MISSING: list[tuple[str, str]] = []

try:
    from harness.runners.run_doctr import DoctrRunner

    _REAL_RUNNERS.append(DoctrRunner())
except ImportError:
    _MISSING.append(("doctr", 'pip install -e ".[doctr]"'))

try:
    from harness.runners.run_paddle_v6 import PaddleV6Runner

    _REAL_RUNNERS.append(PaddleV6Runner())
except ImportError:
    _MISSING.append(("pp-ocrv6_medium", 'pip install -e ".[paddle]"'))

try:
    from harness.runners.run_easyocr import EasyOcrRunner

    _REAL_RUNNERS.append(EasyOcrRunner())
except ImportError:
    _MISSING.append(("easyocr", 'pip install -e ".[easyocr]"'))


def section_phase8() -> None:
    """Show the Runner registry with live versions, then the version guard actually firing."""
    _section("PHASE 8 — the Runner contract: one interface, provenance that travels with data")

    print("Every engine below is a `Runner` subclass. Its `run()` is the ONLY engine-specific")
    print("code; the harness, matcher and scorer never change. `version` is read from the")
    print("installed library at runtime — never hand-typed (CLAUDE.md rule #9).")
    print()

    # A dummy ref purely to exercise the cost rule; no pixels are involved.
    ref = ImageRef(id="_", path="", w=640, h=480, stratum="_", modality="CT", vendor="_",
                   frame_idx=0)
    print(f"  {'model_name':<20}{'version (live)':<18}{'cost/image':<12}pricing rule")
    print(f"  {'-' * 66}")
    for runner in _REAL_RUNNERS:
        # cost() falls back to the self_hosted rule ($0) because these runners are untagged —
        # a fail-safe default, not a guess about the engine.
        cost = runner.cost(ref)
        print(f"  {runner.model_name:<20}{runner.version:<18}${cost:<11.4f}self_hosted (local)")
    for name, how in _MISSING:
        print(f"  {name:<20}{'NOT INSTALLED':<18}{'—':<12}{how}")

    print()
    print("The guard: two result rows that differ ONLY in engine version.")
    print("Before Phase 8 these would have averaged into one number. Now:")
    rows = [
        {"model_name": "doctr", "version": "v1.0.1", "stratum": "s", "n_tokens": 1},
        {"model_name": "doctr", "version": "v1.0.2", "stratum": "s", "n_tokens": 1},
    ]
    try:
        aggregate(rows)
        print("  ⚠️  NO ERROR RAISED — the version guard is not working.")
    except ValueError as exc:
        print(f"  ✅ ValueError: {exc}")
    print()
    print("  An engine version bump invalidates prior results the same way a gt.csv edit does.")
    print("  It can no longer happen silently — the check lives where the blending would.")


# ===========================================================================
# SECTION 2 — PHASE 9: three REAL engines through the one fixed loop
# ===========================================================================

# A tiny synthetic set: two token images across two strata, plus one blank negative control.
# Tokens are fake by construction (tests/synthetic.py's house style) — this is why it is safe
# to run real OCR here and to print what the engines read.
_SET_SPEC = (
    ("demo-000", ("CMFN-0042",), "synth_ct_axial", "FakeVendorA", "CT"),
    ("demo-001", ("ACC-0099", "GRDN1234"), "synth_xr_chest", "FakeVendorB", "XR"),
    ("demo-blank", (), "synth_ct_axial", "FakeVendorA", "CT"),  # negative control
)


def build_real_image_set() -> tuple[list[ImageRef], dict[str, list[GTToken]], set[str]]:
    """Draw the synthetic PNGs to disk and return (images, ground truth, allowlist).

    Real engines load pixels from `ImageRef.path`, so unlike the stub section these images
    must actually exist on disk. They are written under demo/output/images/ — synthetic, and
    already covered by .gitignore's `*.png` rule.
    """
    from tests.synthetic import make_blank_image, make_synthetic_image

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    images: list[ImageRef] = []
    ground_truth: dict[str, list[GTToken]] = {}
    allowlist: set[str] = set()

    for image_id, tokens, stratum, vendor, modality in _SET_SPEC:
        if tokens:
            img, gt = make_synthetic_image(
                tokens, image_id=image_id, stratum=stratum, vendor=vendor,
                modality=modality, label="KEEP",  # KEEP = a valid token we must not destroy
            )
            allowlist.update(tokens)
        else:
            img, gt = make_blank_image()
        path = IMG_DIR / f"{image_id}.png"
        img.save(path)
        w, h = img.size
        images.append(ImageRef(id=image_id, path=str(path), w=w, h=h, stratum=stratum,
                               modality=modality, vendor=vendor, frame_idx=0))
        ground_truth[image_id] = gt

    return images, ground_truth, allowlist


def section_phase9(generated_at: str) -> list[dict]:
    """Run every installed real engine through `run_harness` and report the comparison."""
    _section("PHASE 9 — three REAL local engines, ONE fixed loop, synthetic images")

    if not _REAL_RUNNERS:
        print("No engine installed — nothing to run. See the install hints above.")
        return []

    images, ground_truth, allowlist = build_real_image_set()
    set_hash = _set_hash(images, ground_truth)
    n_tokens = sum(len(v) for v in ground_truth.values())
    print(f"{len(images)} synthetic images ({n_tokens} fake tokens, 2 strata, 1 blank control)")
    print(f"drawn to {IMG_DIR.relative_to(_REPO_ROOT)}/ · allowlist = the {len(allowlist)} KEEP "
          "tokens")
    print()

    summary: list[dict] = []
    for runner in _REAL_RUNNERS:
        print(f"  running {runner.model_name} …", end="", flush=True)
        t0 = time.perf_counter()
        with _quiet():
            # The engine is passed as `runner.run` — a bound method satisfying RunFunc
            # unchanged. Nothing else about this call differs between engines. That is the
            # whole point.
            agg = run_harness(images, runner.run, ground_truth, allowlist=allowlist, iou=0.5)
            first_read = sorted({w.text for w in runner.run(images[0]).words})
        wall = time.perf_counter() - t0

        # One aggregate PER ENGINE — they must not blend, which is exactly section 1's guard.
        out_path = OUT_DIR / f"report_{runner.model_name}.md"
        write_report(agg, out_path, run_metadata={
            "tier": "step-1-local", "version": runner.version,
            "run_hash": hashlib.sha256(
                f"{runner.model_name}|{runner.version}|{set_hash}".encode()).hexdigest()[:12],
            "set_hash": set_hash, "generated_at": generated_at,
        })

        ov, nc = agg["overall"], agg["negative_control"]
        summary.append({
            "name": runner.model_name, "version": runner.version,
            "fr": ov["false_redaction_rate"], "keep_exact": ov["keep_exact_match_rate"],
            "found": ov["found_count"], "omission": ov["omission_count"],
            "added": ov["added_count"], "floor": nc["hallucinations_per_image"],
            "lat": ov["latency"]["mean"] if ov["latency"] else None,
            "cost": ov["cost"]["sum"], "report": out_path, "wall": wall,
            # SAFE ONLY BECAUSE THESE TOKENS ARE SYNTHETIC. On real data this is engine
            # read-back of burned-in PHI and must NEVER be printed (CLAUDE.md §0/§7).
            "read": first_read,
        })
        print(f" {wall:5.1f}s")

    print()
    header = (f"  {'Engine':<18}{'version':<10}{'FR rate':>9}{'KEEP-exact':>12}"
              f"{'Halluc/img':>12}{'Latency':>10}{'Cost':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summary:
        floor = "—" if s["floor"] is None else f"{s['floor']:.2f}"
        lat = "—" if s["lat"] is None else f"{s['lat']:.2f}s"
        cost = "$0" if s["cost"] == 0 else "$" + format(s["cost"], ".4f")
        print(f"  {s['name']:<18}{s['version']:<10}{_fmt_rate(s['fr']):>9}"
              f"{_fmt_rate(s['keep_exact']):>12}{floor:>12}{lat:>10}{cost:>8}")

    print()
    print("  Found vs Added are NEVER averaged — omission axis vs hallucination axis:")
    for s in summary:
        print(f"    {s['name']:<18} Found={s['found']:>2}  Omission={s['omission']:>2}  "
              f"Added={s['added']:>2}")

    print()
    print("  What each engine actually read off the first synthetic image (fake tokens only):")
    for s in summary:
        print(f"    {s['name']:<18} {s['read']}")

    # An engine can read a token perfectly and still score zero Found. Say so explicitly —
    # otherwise the table reads as a recognition failure when it is a localization failure,
    # and the two call for completely different fixes.
    geometry_losses = [s["name"] for s in summary if s["omission"] and s["added"]]
    if geometry_losses:
        print()
        print(f"  ⚠️  {', '.join(geometry_losses)} read tokens correctly (above) yet scored")
        print("      Omission AND Added for the same tokens. That is a BOX-GEOMETRY miss, not a")
        print("      misread: the predicted box fell under the matcher's 0.5 IoU bar, so the")
        print("      read counts as both a miss and an invention. Different failure, different")
        print("      fix — this is exactly why Found and Added are never averaged.")

    print()
    print("  ⚠️  Latency is CPU-only (D-9.1) and is NOT representative of GPU-served")
    print("      production numbers. Do not quote these as throughput figures.")
    return summary


# ===========================================================================
# SECTION 3 — the verifier arm and the cost axis (stub engines)
# ===========================================================================

W, H = 1024, 768


def _img(id_, stratum, modality, vendor, frame_idx):
    # `path` is never opened — the stub engines below fabricate output — but the field is
    # required by ImageRef.
    return ImageRef(id=id_, path=f"synthetic://{id_}", w=W, h=H, stratum=stratum,
                    modality=modality, vendor=vendor, frame_idx=frame_idx)


IMAGES = [
    _img("us01", "us_header", "US", "vendorA", 20),
    _img("ct01", "ct_scout", "CT", "vendorB", 60),
    _img("sc01", "secondary_capture", "OT", "vendorA", 0),
    _img("mg01", "mammo", "MG", "vendorC", 0),
    _img("ct_blank01", "ct_scout", "CT", "vendorB", 60),  # negative control: zero tokens
]


def _gt(image_id, series, modality, vendor, stratum, frame_idx, text, bbox, label):
    return GTToken(image_id=image_id, series_uid=series, modality=modality, vendor=vendor,
                   stratum=stratum, frame_idx=frame_idx, token_text=text, bbox=bbox, label=label)


# Ground truth. KEEP = a valid token we must read & preserve; PHI = must be redacted.
GROUND_TRUTH: dict[str, list[GTToken]] = {
    "us01": [
        _gt("us01", "S-US-1", "US", "vendorA", "us_header", 20,
            "CMFN-00421", (100, 50, 300, 90), "KEEP"),
        _gt("us01", "S-US-1", "US", "vendorA", "us_header", 20,
            "ACC-77812", (100, 120, 320, 160), "KEEP"),
    ],
    "ct01": [
        _gt("ct01", "S-CT-1", "CT", "vendorB", "ct_scout", 60,
            "CMFN-10537", (400, 700, 620, 740), "KEEP"),
    ],
    "sc01": [
        _gt("sc01", "S-SC-1", "OT", "vendorA", "secondary_capture", 0,
            "CMFN-22910", (60, 40, 240, 80), "KEEP"),
        _gt("sc01", "S-SC-1", "OT", "vendorA", "secondary_capture", 0,
            "GRDN-778120", (60, 100, 300, 140), "PHI"),
    ],
    "mg01": [
        _gt("mg01", "S-MG-1", "MG", "vendorC", "mammo", 0,
            "CMFN-33004", (800, 60, 1000, 100), "KEEP"),
    ],
    "ct_blank01": [],  # confirmed-blank negative control
}

# Allowlist = the known-good KEEP values. A token is preserved only if OCR reads it exactly
# AND it is on this list; PHI codes are deliberately absent so PHI is redacted.
ALLOWLIST = {"CMFN-00421", "ACC-77812", "CMFN-10537", "CMFN-22910", "CMFN-33004"}

# Boxes reused by the stub engines (perfect localization; the stubs vary only text/conf).
_BOX = {
    ("us01", 0): (100, 50, 300, 90), ("us01", 1): (100, 120, 320, 160),
    ("ct01", 0): (400, 700, 620, 740),
    ("sc01", 0): (60, 40, 240, 80), ("sc01", 1): (60, 100, 300, 140),
    ("mg01", 0): (800, 60, 1000, 100),
}


def _out(words, model_name, version, config_id="demo"):
    # raw_response is opaque + PHI-bearing by contract; here it's a synthetic placeholder
    # that is never printed or logged (demonstrating the "never echo raw_response" rule).
    #
    # config_id/config_hash are required identity fields (D-13.5). These stub engines have no
    # real config, so the hash is derived from the demo engine's own name — enough to give
    # each stub a distinct identity, which is what aggregate() checks. A real runner derives
    # its hash from `Runner.config()` instead of doing this.
    config_hash = hashlib.sha256(f"{model_name}:{config_id}".encode()).hexdigest()[:12]
    return OCROutput(words=words, raw_response={"synthetic": True}, model_name=model_name,
                     version=version, config_id=config_id, config_hash=config_hash)


@priced("cloud")  # cloud OCR -> flat $/image, to show the cost axis
def cloud_ocr_demo(img: ImageRef) -> OCROutput:
    """Sloppier reader: one misread (low conf), one omission, hallucinations on the blank."""
    time.sleep(0.006)  # stand-in for real inference latency (timer wraps only this call)
    reads = {
        # ACC-77812 misread as ACC-7781Z at LOW confidence -> a false redaction the
        # verifier can later recover.
        "us01": [("CMFN-00421", 0, 0.91), ("ACC-7781Z", 1, 0.40)],
        "ct01": [],  # token omitted entirely -> false redaction the verifier CANNOT recover
        "sc01": [("CMFN-22910", 0, 0.88), ("GRDN-778120", 1, 0.83)],
        "mg01": [("CMFN-33004", 0, 0.92)],
        # blank frame: two invented boxes -> the hallucination floor (high conf, not re-read)
        "ct_blank01": [
            OCRWord(text="CMFN-99999", bbox=(200, 200, 380, 240), confidence=0.90),
            OCRWord(text="ACC-00000", bbox=(500, 500, 700, 540), confidence=0.92),
        ],
    }
    entry = reads[img.id]
    words = entry if entry and isinstance(entry[0], OCRWord) else [
        OCRWord(text=t, bbox=_BOX[(img.id, i)], confidence=c) for (t, i, c) in entry
    ]
    return _out(words, "cloud-ocr-demo", "cloud-ocr-2026.07-demo")


def demo_verifier(img: ImageRef, word: OCRWord) -> str:
    """Fake verifier arm: re-reads a low-confidence word and returns the corrected string.

    The real verifier (`apply_verifier`) runs ONLY on words below the confidence threshold,
    keeps the primary's box, and swaps only the text. Here the re-read fixes the ACC misread;
    it has no token to recover for the omitted ct01 word.
    """
    time.sleep(0.004)  # verifier time is measured separately from the primary model time
    return "ACC-77812" if word.text == "ACC-7781Z" else word.text


_STUB_CONFIGS = [
    {"key": "cloud", "label": "cloud-ocr-demo (priced)", "verifier": False},
    {"key": "cloud_verified", "label": "cloud-ocr-demo + verifier arm", "verifier": True},
]


def section_verifier(set_hash: str, generated_at: str) -> None:
    """The two things real local engines can't show: a priced engine and a verifier recovery."""
    _section("THE VERIFIER ARM AND THE COST AXIS — stub engines")

    print("All three local engines are $0 and emit their own boxes, so a stub cloud engine is")
    print("the only way to show the cost axis and a verifier recovering a low-confidence read.")
    print()

    results = []
    for cfg in _STUB_CONFIGS:
        kwargs = {
            "verifier_func": demo_verifier,
            "verifier_model_name": "qwen3-vl-demo",
            "verifier_version": "demo-2026.07",
            "conf_threshold": 0.60,
        } if cfg["verifier"] else {}
        agg = run_harness(IMAGES, cloud_ocr_demo, GROUND_TRUTH, allowlist=ALLOWLIST, iou=0.5,
                          **kwargs)
        out_path = OUT_DIR / f"report_{cfg['key']}.md"
        write_report(agg, out_path, run_metadata={
            "tier": "step-4-cloud-fallback" + ("+verifier" if cfg["verifier"] else ""),
            "version": "cloud-ocr-2026.07-demo",
            "run_hash": hashlib.sha256(
                f"{cfg['key']}|{set_hash}".encode()).hexdigest()[:12],
            "set_hash": set_hash, "generated_at": generated_at,
        })
        ov = agg["overall"]
        results.append({"label": cfg["label"], "fr": ov["false_redaction_rate"],
                        "keep_exact": ov["keep_exact_match_rate"], "cost": ov["cost"]["sum"],
                        "report": out_path})

    header = f"  {'Config':<34}{'FR rate':>9}{'KEEP-exact':>12}{'Cost':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['label']:<34}{_fmt_rate(r['fr']):>9}{_fmt_rate(r['keep_exact']):>12}"
              f"{'$' + format(r['cost'], '.4f'):>10}")
    print()
    print("  The verifier recovers the low-confidence misread but CANNOT recover an outright")
    print("  omission — a missed token has no low-confidence word to re-read.")


# ===========================================================================


def _set_hash(images: list[ImageRef], ground_truth: dict[str, list[GTToken]]) -> str:
    """A stable hash of the (synthetic) scored set — mirrors the real run-metadata hash."""
    material = "|".join(f"{i.id}:{i.stratum}:{len(ground_truth[i.id])}" for i in images)
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip the real-engine section (no model loading)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    started = time.perf_counter()

    print(RULE)
    print("OCR benchmark harness — Phases 8 & 9 walkthrough (SYNTHETIC data, PHI-free)")
    print(RULE)

    section_phase8()

    reports: list[Path] = []
    if args.fast:
        _section("PHASE 9 — skipped (--fast)")
        print("Real engines not loaded. Drop --fast to run the live three-engine bake-off.")
    else:
        real_summary = section_phase9(generated_at)
        reports += [s["report"] for s in real_summary]

    section_verifier(_set_hash(IMAGES, GROUND_TRUTH), generated_at)
    reports += [OUT_DIR / f"report_{c['key']}.md" for c in _STUB_CONFIGS]

    _section("REPORTS")
    for path in reports:
        print(f"  {path.relative_to(_REPO_ROOT)}")
    print()
    print(f"Total: {time.perf_counter() - started:.1f}s")
    print("Takeaway: ONE fixed scoring loop ranks every engine, and each result now carries the")
    print("exact engine version that produced it — so a version bump can't silently blend into")
    print("prior numbers. (All synthetic: no DICOM, no gt.csv, no network.)")


if __name__ == "__main__":
    main()
