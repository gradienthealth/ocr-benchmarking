# 13i — the real pass: what Arnav runs, in test-plan order

**Claude built this; a human runs it (D-12.3).** Every command below opens real renders, and the
Gemini block ships crops to a vendor — the sandbox's egress allowlist excludes `googleapis` on
purpose, so the agent cannot be the thing that does it.

Every command is **one line**. No backslash continuations: a wrapped paste mangles the flags into
literal escaped spaces.

Paths assume the `v2` render set (199 images = `gt_v1`). Substitute your own render dir if it
differs — the scored-set hash check will tell you immediately if it is the wrong set.

Each command writes to a **new** `--out` directory. That is enforced: a non-empty output dir is
refused, because a stale `TABLES.md` sitting beside fresh arm artifacts reads as a complete run.
If a run dies halfway it writes `run_status.json: INCOMPLETE` and **no table at all** — re-run it
into a new directory.

---

## Step 0 — prove the wiring with no PHI and no network (~15 s)

```
.venv/bin/python scripts/run_experiment.py --synthetic --all-arms --out results/00_synthetic_smoke
```

Green here means the CLI → three-tables path works. It says nothing about any engine.

## Step 0b — the hash gate, on the real frozen artifact (instant, opens no pixel)

```
.venv/bin/python scripts/run_experiment.py --verify-only --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv
```

Expected: `gt.csv sha256 afc65989…` and `scored-set sha256 e0bf9503…`, 199 images, 97 blank
control. Any mismatch aborts every command below — that is the point.

---

## Step 1 + 2 — bake-off at stock config, negative control included

**The negative control is not a separate command for these arms and does not need to be.** An
end-to-end engine invents text by drawing a box where there is none, so the 97 confirmed-blank
frames in the scored set *are* its negative control, measured in the same pass and printed under
Table A before any ranking is quoted from it.

```
.venv/bin/python scripts/run_experiment.py --arm doctr:stock --arm pp-ocrv6_medium:stock --arm easyocr:stock --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/01_bakeoff_stock
```

Read Table A for the ranking, Table B for whether a loss was a *detection* loss, and the floor
notes under Table A before believing either. EasyOCR is the **floor, not a contender** — it is in
the table so the ranking has a bottom, not so it can win.

## Step 6 — stock vs tuned (feeds step 1; the bake-off ranking is meaningless without it)

```
.venv/bin/python scripts/run_experiment.py --arm doctr:tuned --arm pp-ocrv6_medium:tuned --arm easyocr:tuned --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/02_bakeoff_tuned
```

⚠️ **`doctr:tuned` and `pp-ocrv6_medium:tuned` will abort with `ArmNotFrozenError` today.** Their
thresholds are an *output* of the 13h dev-slice sweep, which has not been completed —
`experiments/tuned_configs.json` has no entry for them. That abort is correct: an arm that
silently fell back to stock values while calling itself "tuned" would put a mislabelled row
straight into the report. `easyocr:tuned` is buildable now (both ends of its pair are already
known, no sweep needed), so run it alone if you want the one pair that exists:

```
.venv/bin/python scripts/run_experiment.py --arm easyocr:stock --arm easyocr:tuned --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/03_easyocr_pair
```

## Step 7 — the reading arm (oracle GT boxes, Table C only)

The floor runs **first, automatically, every time** — there is no flag that skips it. Table C's
accuracy columns cannot be produced without it.

```
.venv/bin/python scripts/run_experiment.py --arm reader:svtrv2 --arm reader:doctr-parseq --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/04_readers_local
```

Qwen3-VL gets its own command — it is generative, it is slow, and it needs its own venv (never
the shared benchmark one; a pip install from a worktree lands in the environment docTR/PP-OCRv6/
EasyOCR are being measured in, and that invalidates comparability the way an engine bump does):

```
.venv/bin/python scripts/run_experiment.py --arm reader:qwen3vl --boxes-per-image 4 --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/05_reader_qwen3vl
```

**Check the clock before starting this one.** Cost is per crop: 97 blank frames × 4 control boxes
= 388 floor crops, then 2,351 GT boxes. At the measured ~26 s/crop on an idle 8-core box
(`experiments/qwen3vl_cpu_timing.py`) that is **~2.8 h for the floor and ~17 h for accuracy**.
Nothing checkpoints. Launch it detached, or cut the scope deliberately and say in the write-up
that you did.

---

## The Gemini arm — its own block, because it is its own category of thing

Every crop here **leaves the environment**. Gradient's Google BAA is signed and model usage
**within Vertex AI** is acceptable (confirmed in writing 2026-08-11); the Gemini Developer API /
AI Studio (`GEMINI_API_KEY`) is a different product and is **not** covered. The driver enforces
Vertex-only and aborts before the first crop if the project or location is missing.

### G1 — prove the plumbing on fake pixels first. No clearance needed, costs cents.

```
.venv/bin/python experiments/bench_reader_synthetic.py --reader gemini --out results/06_gemini_synth.json
```

Fake `CMFN`-style crops, registered by digest, over a live Vertex call. **If the pinned model id
is stale or the project is misconfigured, this is where it should surface — not on frame 40 of a
real run.** Note that changing `PINNED_MODEL` is an engine bump: re-run, don't mix.

### G2 — the two lines a human types. Neither is in this repo, and neither should be.

```
export GOOGLE_CLOUD_PROJECT=gradient-health-central
```
```
export GOOGLE_CLOUD_LOCATION=us-central1
```
```
export OCR_BAA_CLEARED_GEMINI=1
```

That last line is the gate. **Nothing in this repo sets it** — not the driver, not a `.env`, not
a Makefile, and a test greps the driver to keep it that way. Set it only after the D-12.1
checklist in `prompts/phase13/13g_RESULT.md` §4 is confirmed, in your own shell, for one session.
The reader re-checks it above the transport before **every single crop**, so closing the terminal
closes the gate.

### G3 — the run.

```
.venv/bin/python scripts/run_experiment.py --arm gemini:real --boxes-per-image 4 --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --text-presence ground_truth/text_presence_v2.csv --out results/07_gemini_real
```

`gemini:real` is never selected by `--all-arms` and cannot be reached with `--synthetic`. It has
to be named, exactly like this, every time.

What to expect and what to check afterwards:

| | |
|---|---|
| Crops that egress | 388 control + 2,351 GT = **2,739**, each a burned-in identifier |
| Cost | order of **$2–4**, dominated by billed thinking tokens. Report `measured_cost_usd` from the run metadata, **not** `harness/cost.py`'s estimate — the estimate prices the input side from crop dimensions and cannot see output, and 2.5 Pro's thinking cannot be switched off |
| Version | `served_model_version` is stamped from the *response*. Vertex serves an alias, so the run date is part of the arm's identity |
| If it dies | it dies. There is **no retry at any layer** — a retry is a second egress event on the same PHI. A `ServedVersionMismatch` (the alias re-pointed mid-run) kills the run by design; half a table from one revision and half from another is the silent engine bump rule #9 exists to stop. Re-run into a new dir |
| What is written | aggregates only. No `raw_response`, no predicted string, no crop — on this arm the predicted string *is* the patient identifier read back |

---

## After every run

- `results/<dir>/run_status.json` — `COMPLETE` or `INCOMPLETE`, and which arm failed.
- `results/<dir>/TABLES.md` — the three tables, stamped with `gt_v1` + both hashes + 199 images.
  Exists **only** for a complete run.
- `results/<dir>/arms/<arm>/report.md` — the full Phase-7 stratified report (end-to-end arms).
- `results/` is gitignored. Commit hashes and findings, never the artifacts.

Assembling one findings narrative across these runs is **13l**, not this step. Each command's
`TABLES.md` covers exactly the arms named in that command — say which arms were skipped rather
than letting a partial roster read as the whole comparison.
