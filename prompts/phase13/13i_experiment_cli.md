# 13i — `scripts/run_experiment.py`: the driver 🟡 build PHI-free, run is PHI-touching

> **Prerequisites: 13a, 13b, 13e, and at least one reader (13c). Also 13g, if the Gemini arm is in
> scope** — it is the only BAA-gated arm the driver can select, and it changes what "abort" means.
> This is the thin CLI that runs the test plan end to end; it wires existing pieces and adds no
> scoring logic of its own.

---

## Goal

One command that selects arms, verifies the frozen `gt.csv` hash, runs the harness, and writes the
three report tables — with PHI-free aggregates only on stdout.

## Context

- plan.md **PHASE 13** — the six-step test plan this executes, in order.
- `harness/harness.py` — `run_harness()`; `harness/aggregate.py`; `harness/report.py`.
- `harness/reading.py` — the reader arm path (13b).
- `ground_truth/gt.csv.sha256` = `afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e`
  and `ground_truth/gt_set.sha256` = `e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea`
  — `gt_v1`, 199 images, 2351 rows.
- `ground_truth/gt_schema.py` — `load_gt()`; a blank image contributes zero rows.
- `harness/readers/read_gemini.py` (13g) — the BAA-gated Gemini reader, plus
  `prompts/phase13/13g_RESULT.md` §4, the D-12.1 checklist it is gated on. Gradient's Google BAA is
  signed and **model usage within Vertex AI is acceptable** (confirmed in writing 2026-08-11), so this
  arm is cleared to read real `gt_v1` frames once a human sets the flag. It is still the only arm in
  the driver whose every crop leaves the environment — treat it as a different category of thing, not
  as a seventh row.

## Constraints

- **Verify the `gt.csv` sha256 first and abort on mismatch.** A changed `gt.csv` invalidates every
  previously reported number (rule #8); the driver must refuse to run rather than quietly produce
  incomparable results. Print the hash it verified.
- **Stamp every result with the artifact identity** — `gt_v1`, both hashes, image count — so no table
  can be mistaken for a future 307- or 966-image build.
- **PHI-free stdout only**: counts, rates, hashes, shapes. Never a token, never a path that reveals
  a UID, never a crop.
- **Iteration order is free** — metrics are order-independent means (D-1.1). Do not add an ordering
  requirement or a "never re-sort" rule.
- **Three tables, never blended**: end-to-end (Arm A), detector-only (Arm B), reader (Arm C). Arm C
  gets oracle boxes and is not comparable to Arm A.
- **Do NOT modify** `matching.py`, `metrics.py`, `normalize()`, or any runner. This is wiring.
- Every generative arm runs the **negative control before its accuracy numbers are reported**, and the
  driver should make that ordering hard to skip.
- Resume/partial-run behaviour: if a run dies halfway, it must not emit a table that looks complete.

### The Gemini arm on real `gt_v1` frames (13g) — extra constraints, all of them load-bearing

- **Never construct a `REAL` Gemini reader implicitly.** Selecting "all arms" must not select it. It
  takes its own explicit selector (e.g. `--arm gemini:real`), and the driver **never sets
  `OCR_BAA_CLEARED_GEMINI` itself** — not in code, not in a `.env`, not in a Makefile. If the selector
  is given and the variable is absent, abort with the D-12.1 pointer, the same way a hash mismatch
  aborts. A cleared human sets it in their shell, for one session, and the reader re-checks it before
  every single crop.
- **Vertex only.** `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` must be present or the run aborts
  before the first crop, with the reason named: the BAA covers model usage **within Vertex AI**, and
  the AI Studio `GEMINI_API_KEY` surface is a different, uncovered product.
- **Negative control first, and make skipping it structurally hard for this arm specifically.** It is
  generative, so its hallucination floor on the 64 blank `ct_axial` frames must be measured and
  printed *before* any accuracy number for it is emitted. The reader arm's only route to a floor is
  `control_boxes` on confirmed-blank frames (`read_image` refuses them on text-bearing images, and
  that refusal is deliberate) — so the driver has to supply them, and should refuse to print the arm's
  accuracy table at all if the floor step did not run.
- **Report measured cost, not the estimate.** `harness/cost.py` prices the input side from crop
  dimensions and cannot see output; on 2.5 Pro, thinking tokens are billed as output and cannot be
  switched off. Pull `reader.usage_summary()` into the run metadata and report
  `measured_cost_usd` as this arm's cost, with the estimate shown beside it if you show it at all.
- **Record the served revision, and let `ServedVersionMismatch` kill the run.** The reader raises when
  the revision that answered differs from the pinned one; do not catch it, do not retry, do not
  downgrade it to a warning. Half a table from one revision and half from another is exactly the
  silent engine bump rule #9 exists to stop. Stamp `served_model_version` next to the `gt_v1` identity.
- **No retry loop, at any layer.** The reader deliberately does not retry: a retry is a second egress
  event on the same PHI, and it makes cost and latency depend on the network's mood. The driver must
  not add one back. A cloud arm dying on a 429 halfway through is exactly the partial-run case above —
  emit no complete-looking table.
- **Nothing writes model output.** The driver persists aggregates; no `raw_response`, no per-crop
  predicted string, no debug dump. On this arm the predicted string *is* the patient identifier read
  back (CLAUDE.md §3).
- **Claude builds it, a human runs it** (D-12.3). The deliverable for the real pass is the command,
  not an executed run — the sandbox's egress allowlist excludes `googleapis` on purpose, and the
  agent must not be the thing that ships PHI to a vendor.
- **Prove the plumbing on synthetic crops before spending PHI egress.** `experiments/bench_reader_synthetic.py --reader gemini` makes a live Vertex call with fake `CMFN` pixels and needs no clearance. If the pinned model id is stale or the project is misconfigured, that is where it should surface — not on frame 40 of a real run. Note that changing `PINNED_MODEL` is an engine bump: re-run, don't mix.

## Steps

1. Enter **Plan Mode**: enumerate the CLI surface (flags, arm selection, output paths), the abort
   conditions, and the exact table set. Order of operations matters here — state it.
2. Implement `scripts/run_experiment.py` + tests over synthetic fixtures, including: hash mismatch
   aborts, a partial run does not emit a complete-looking table, and arm selection produces exactly
   the tables selected.
3. Cover the Gemini arm in the same test file, on synthetic fixtures and with a stub transport — no
   live call in `pytest`, ever. At minimum: selecting `gemini:real` without `OCR_BAA_CLEARED_GEMINI`
   aborts; "all arms" does not select it; missing Vertex project/location aborts; a
   `ServedVersionMismatch` propagates instead of being swallowed; the accuracy table is refused when
   the negative-control step did not run.
4. Write the command list Arnav runs for the real pass, in test-plan order, with the negative control
   first. Single-line commands, no backslash continuations. The Gemini arm gets its own block,
   preceded by the synthetic bench and by the one line that sets the gate variable — written out so
   it is obvious that a human types it.

## Output format

Plan first. Then the diff, test output, and a sample of each of the three tables rendered from
synthetic data.

## Done when

- `pytest` green; `ruff check .` clean.
- A deliberately wrong hash aborts the run with a clear message.
- Sample tables show the `gt_v1` identity stamp.
- `git diff --stat` shows no change to `matching.py`, `metrics.py`, or `contract.py`.
- A grep shows the driver never sets `OCR_BAA_CLEARED_GEMINI`, and selecting every arm does not
  select `gemini:real`.
- The Gemini arm's run metadata carries `served_model_version` and `measured_cost_usd`.

## Finally

Spawn one fresh-context subagent. Scope: can any code path print token text or a UID-bearing path,
can a run proceed past a hash mismatch, can a partial run produce a table that reads as complete, and
can a real crop reach Gemini without a human having set `OCR_BAA_CLEARED_GEMINI` — including via arm
selection, a config file, a fixture, or an error handler? Correctness only; report only paths it can
actually construct.
