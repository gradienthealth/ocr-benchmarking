# Harness demo (synthetic, PHI-free)

A one-command walkthrough of what Phases 8 and 9 produced. **Everything is synthetic** — images
drawn by `tests/synthetic.py` with fake tokens, plus stub engines for the cloud/verifier arm. No
DICOM, no `gt.csv`, no GCS, no network egress. Safe to run and to show.

## Run it

```
python demo/run_demo.py          # full walkthrough, ~40s (loads three real OCR models)
python demo/run_demo.py --fast   # skips the real-engine section, ~10s
```

Prints a console walkthrough and writes five markdown reports + charts to `demo/output/`.
Open the `.md` files in any markdown viewer to see the stratified tables and charts.

Models are cached after the first run (`~/.cache/doctr`, `~/.paddlex`, `~/.EasyOCR`), so nothing
downloads mid-demo — but do one warm-up run before presenting.

## What it shows

### 1. Phase 8 — the `Runner` contract and version provenance

Each engine is a `Runner` subclass whose `run()` is the only engine-specific code. The table
shows each engine's `model_name`, its `version` **read live from the installed library**, and
its cost rule ($0, from the untagged→`self_hosted` fallback).

Then the guard fires: two result rows differing *only* in engine version are handed to
`aggregate()`, which raises rather than averaging them. An engine version bump invalidates
prior results the same way a `gt.csv` edit does, and it can no longer happen silently.

### 2. Phase 9 — three real local engines, one fixed loop

docTR, PP-OCRv6_medium, and EasyOCR each read real synthetic PNGs and are scored by the
**identical** matcher and scorer:

```
OCROutput contract  ->  IoU matching  ->  per-image scoring  ->  aggregate  ->  report + charts
   contract.py           matching.py        metrics.py          aggregate.py     report.py
```

| Engine | Role |
| --- | --- |
| `doctr` | Step-1 incumbent |
| `pp-ocrv6_medium` | Step-1 bake-off contender |
| `easyocr` | deliberate floor/baseline — detection threshold 0.2, not a contender |

Watch for the EasyOCR row: it reads the tokens correctly yet scores zero Found. Its boxes run
~1.9× taller than the glyph content, so they fall under the matcher's 0.5 IoU bar and each read
counts as an omission **and** a hallucination — a localization failure, not a recognition one.
The demo calls this out explicitly, and it is the clearest possible argument for why Found and
Added are never averaged into one score.

Latency shown is **CPU-only** (D-9.1) and is not representative of GPU-served production.

### 3. The verifier arm and the cost axis (stub engines)

All three local engines are $0 and emit their own boxes, so stubs remain the only way to show a
priced cloud engine and a verifier recovering a low-confidence misread — and failing to recover
an outright omission, which has no low-confidence word to re-read.

## Headline ideas on display

**False-redaction rate** as the headline metric, **Found vs Added as separate axes** (never
averaged), per-stratum breakout, the **negative-control hallucination floor**, primary-vs-
verifier latency split, per-engine cost, per-row engine-version provenance, and CER/WER
quarantined in a diagnostic appendix.

`demo/output/` is regenerated on every run and is safe to delete.
