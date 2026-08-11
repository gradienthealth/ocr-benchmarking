# 13c — Arm C readers: SVTRv2 + docTR-parseq 🟢 PHI-free (synthetic test only)

> **Prerequisites: 13a (config identity) and 13b (`harness/reading.py`) merged.**
> Two readers in one session because they share the same interface and the same test scaffold.
> Build and test on synthetic fixtures only — Arnav runs anything touching real renders.

---

## Goal

Two working readers on the 13b path — SVTRv2 and docTR's `parseq` head — each declaring a distinct
config identity, each proven on synthetic crops.

## Context

- `harness/reading.py` — the reader interface and the pinned crop preprocessing (from 13b).
- `harness/runners/run_doctr.py` — the existing docTR runner; the parseq reader reuses its weight
  handling and `version_source` pattern.
- `openocr-python` **0.1.5 is already installed** in `.venv`. Entry point: `OpenOCR(task='rec')`.
  Recognition postprocess returns `(text, mean_char_probability)` —
  `.venv/lib/python3.12/site-packages/openocr/openrec/postprocess/ctc_postprocess.py:84`.
- docTR parseq: `ocr_predictor(reco_arch='parseq', pretrained=True)`, or the recognition predictor
  alone if only the reco head is needed for crop-in use.
- CLAUDE.md §4 — record engine name + tier + **exact version string** in run metadata (rule #9).

## Constraints

- **`drop_score=0.0` on the OpenOCR reader — non-negotiable.** It defaults to **0.5** and silently
  discards low-confidence reads before you ever see them. That is a seed-time filter, which
  CLAUDE.md §8 forbids ("filter at view time, not seed time"). A discarded read must still reach the
  scorer as a read.
- **Suppress OpenOCR's default result write.** It writes `./e2e_results/system_results.txt` into the
  CWD; on real data that is engine-read token text landing in an unmanaged file. Verify no file is
  created anywhere outside a path the caller specifies.
- **Neither reader may alter the crop preprocessing** pinned in 13b.
- **Confirm each reader's charset covers `A-Z0-9-`** before scoring, and add a test asserting it.
  Several recognizers ship Chinese-first dictionaries and would fail for reasons that have nothing to
  do with reading quality.
- Versions sourced from the installed libraries, never hand-typed. Each reader declares its own
  config identity per 13a — docTR-parseq and docTR-stock must not collide.
- **Do NOT touch** `harness/reading.py`'s scoring, `matching.py`, `metrics.py`, or `normalize()`.
- These are **readers**, not engines: neither goes in the end-to-end arm. OpenOCR's *detector* is
  ruled out — see plan.md PHASE 9 "Ruled out" for the measured reason; do not re-litigate or try to
  wire it.

## Steps

1. Build the SVTRv2 reader first, complete with tests. Then the docTR-parseq reader.
2. For each: a determinism test (same crop twice → identical output) and a known-token test (a
   synthetic `CMFN-00421` crop reads back exactly).
3. Record the resolved version strings for both in the session output so they can go into run
   metadata.

## Output format

Diff per reader, one at a time, with test output after each. Do not batch both then test once.

## Done when

- `pytest` green including: charset coverage, determinism, known-token read, and a no-stray-file
  assertion for OpenOCR.
- `ruff check .` clean.
- `git status` shows no `e2e_results/` or other stray output directory.
- The two readers report distinct config identities under 13a's scheme.

## Finally

Spawn one fresh-context subagent to review both readers. Scope: is `drop_score` actually 0.0 on every
code path, can either reader write a file the caller did not ask for, and does either mutate the
shared crop preprocessing? Correctness and constraint violations only.
