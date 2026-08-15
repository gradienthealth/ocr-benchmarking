# 13d RESULT — Qwen3-VL-4B-Instruct reader (Arm C), self-hosted

Branch `13d`, merged with `main` (13c's two recognizer readers). 🟢 Everything below is
PHI-free: every crop measured here was drawn in-memory from fake tokens by `tests/synthetic.py`.

## What shipped

| File | What it is |
| --- | --- |
| `harness/readers/read_qwen3vl.py` | `Qwen3VLReader` — one crop in, one string out, in-process |
| `tests/test_reader_qwen3vl.py` | 50 weight-free contract tests + 4 weight-backed (`QWEN3VL_WEIGHTS=1`) |
| `tests/test_reader_negative_control.py` | 16 tests for the floor procedure's geometry and counting |
| `experiments/qwen3vl_cpu_timing.py` | 🟢 CPU latency on synthetic crops (Claude may run it) |
| `experiments/reader_negative_control.py` | 🔴 the hallucination floor — **Arnav runs this** |
| `tests/test_readers.py` | 13c's shared scaffold, extended: this reader joins it under `QWEN3VL_WEIGHTS=1` |

Pinned: `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`,
transformers 5.15.0, torch 2.13.0+cpu, fp32, greedy, 8 threads. Model revision, torch version,
dtype, thread count and the full prompt text are all in `config()` and therefore in
`config_hash` — a change to any of them makes a new arm that `aggregate()` will not blend.

## CPU timing (n=12 synthetic crops, 1 warm-up, fp32, 8 threads)

| | seconds per crop |
| --- | --- |
| mean | **45.9** |
| median | **45.3** |
| p95 | 58.1 |
| min / max | 36.7 / 58.1 |
| model load | 43.7 s |

**Read that as an upper bound, not the number.** The run shared the box with 13c's and 13g's
sessions (load average ~6 on 8 cores). A single crop measured on an idle box earlier in the
session took **26.0 s**, and the model loaded in 27.8 s. The true idle figure is somewhere
around 26-30 s/crop; the script now records `load_average_1min_at_start/_at_end` in its JSON so
this can never be ambiguous again, but **a clean idle re-measure is still outstanding** — two
attempts at one were killed (see below).

Quality signals from the same run, both SYNTHETIC and neither quotable as accuracy: 12/12 exact
matches, 0 chatty outputs, 0 false abstentions.

### ⚠️ fp32 4B is ~16 GB resident and got OOM-killed twice (exit 137)

Both attempted re-measures died mid-load while other sessions held models on this 29 GB box.
Consequences for the real work:

- The negative-control run holds that 16 GB for its whole duration (hours). Run it when nothing
  else large is running, or it will die partway.
- If memory is the binding constraint, `dtype="bfloat16"` halves it to ~8 GB. That is a
  **different arm** — dtype is in `config_hash`, so results across the two must not be pooled,
  which is the correct and loud behaviour, not a workaround to apply quietly.
- This box is avx2-only (no avx512_bf16, no AMX), so bf16 buys memory, not speed.

## Planning arithmetic for whoever schedules Arm C

Cost is per **crop**, not per image. At ~26 s/crop idle:

- one 30-token ultrasound frame ≈ 13 minutes
- `gt_v1`'s text-bearing side, at ~5 tokens/image over 133 images ≈ 4.8 hours
- the negative control at 4 boxes over the 97 blank frames ≈ 2.8 hours

Any plan that assumes this arm runs like docTR is wrong by two orders of magnitude.

## The negative control — the command Arnav runs

Run this **before** quoting any accuracy number for this arm (plan.md's ordering rule for
generative arms). It writes counts, rates, hashes and identities only — never a token string,
never a pixel.

```
.venv/bin/python experiments/reader_negative_control.py --renders renders/v2 --text-presence ground_truth/text_presence_v2.csv --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --gt ground_truth/gt.csv --out experiments/qwen3vl_negative_control
```

Notes on that command:

- It needs a venv with the `[qwen3vl]` extra. **This worktree's venv is deleted** — recreate one
  (never the shared benchmark venv): `python3 -m venv .venv-qwen`, then CPU torch first, then
  `pip install -e ".[qwen3vl]"`. The pyproject comment block has the exact order.
- `--gt` is read for two non-scoring things only: the set of `image_id`s that have tokens, and
  the median token box height per stratum. `token_text` is never read.
- The blank set is **97** frames (`ct_axial` + `ct_scout` + `mg_tomo`), not the 64 the prompt
  assumed — 64 is the `ct_axial` count alone. Quote the per-stratum breakdown, not one number.
- `--boxes-per-image` defaults to 4 (corners first, where burned-in text actually lives) and
  goes up to 9. Higher N is a superset of lower N, so a follow-up extends rather than replaces.
- The floor it reports is an **upper bound**: a free-form refusal ("There is no readable text.")
  is scored as a read. Only the marker and its punctuated near-misses count as abstention,
  because matching arbitrary prose could hide a real invention.

## Review

One fresh-context subagent reviewed the diff (egress, one-crop-only, abstain contract,
logprob-as-detection-confidence, identity, floor correctness). It cleared the three hard
constraints — no egress path, one crop per call, no confidence extractable — and found four
MAJORs, all fixed in this branch:

1. `model_name` was hardcoded while `model_id` is the D-11.1 A/B knob → an 8B run would report
   under the 4B's name. Now derived from `model_id`.
2. Abstention was exact-string only, so `<<NOTEXT>>.` / `"<<NOTEXT>>"` / `<< NOTEXT >>` counted
   as reads — and on a control box a read is an **invention**, inflating the floor.
3. The control-box spec is outside `arm_config_hash`, so a 4-box floor run, a 9-box run and a
   real accuracy run shared one identity tuple. Floor runs now take `config_id="floor-Nbox"`.
4. Passing `{}` as ground truth made `read_image`'s zero-GT guard unable to fire. `--gt` is now
   required and `guard_presence_matches_gt()` refuses any frame marked blank that has tokens.

Minors fixed: blank-set count (97, not 64) and the runtime estimate; control boxes sized from
measured GT token heights instead of 3.5% of the frame (which *downscaled* on mammo renders
while every real crop upscales); p95 now reuses the harness's `_latency_stats`; `torch` version
joined the identity; a processor-only test proves the chat template emits exactly one vision
placeholder — without it, a template that dropped the image would make every read invention
while all fake-processor tests still passed.

## Merge notes (13c ↔ 13d)

- 13c and 13d independently chose `harness/readers/read_*.py`; the package `__init__.py` came
  out byte-identical, so only `pyproject.toml` conflicted (two extras added at the same point).
  Both `openocr` and `qwen3vl` are present.
- This reader now participates in 13c's **shared** reader checks (contract, version sourcing,
  charset, determinism, known token) rather than only its own — but **opt-in** via
  `QWEN3VL_WEIGHTS=1`, because the shared tier reads three crops per reader and this one is
  16 GB and ~40 s per crop. `charset()` is implemented for it: vocabulary entries that decode
  to exactly one character.
- Fixed a collection crash in `tests/test_readers.py`: with no reader installed, pytest passes
  its NOTSET sentinel to the `ids=` function, which raised and ERRORed the entire file instead
  of skipping it — taking `test_every_reader_is_registered`, the test whose whole job is to
  notice missing readers, down with it.

## Outstanding

- **Idle CPU re-measure** (OOM-killed twice under contention).
- **The negative control has not been run** — it is PHI-touching and Arnav's to run. No accuracy
  number for this arm may be reported before it is.
- `harness/reading.py`'s remark that "the blank control set is all `ct_axial`" is now out of
  date; the presence file has three blank strata.
- 13c installed `openocr-python` into the **shared** benchmark venv. That is the environment
  docTR/PP-OCRv6/EasyOCR are measured in (CLAUDE.md rule #9 / the worktree-venv note) — worth a
  look before the next scored run, though nothing here depends on it.
