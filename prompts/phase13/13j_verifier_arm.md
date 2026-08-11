# 13j — Verifier arm + the D-11.1 A/B 🟡 build PHI-free, run is PHI-touching

> **Was `prompts/phase11/11_verifier_arm.md`; renumbered into the Phase-13 sequence 2026-08-10.**
> The phase numbers were never chronological — plan.md PHASE 13's test plan invokes this as its
> **step 3**, so it belongs after the driver (13i) and before the report (13l), not before Phase 13.
> plan.md's own **PHASE 11** section is unchanged and remains the spec this implements.
>
> **Prerequisites: 13a (config identity) and 13d (Qwen3-VL-4B reader) done**, plus 13i's bake-off run
> for the primary this gates on. 13d already measured crop-reading accuracy and CPU cost for the
> leading candidate — this phase turns the winner into a *gated verifier* and measures whether gating
> helps at all.
>
> **Self-hosted only. Weights stay in-environment.** Arnav runs anything touching real crops.

---

## Goal

A verifier that re-reads only low-confidence crops, plus the measurement that says whether it is worth
having: primary-only vs primary+verifier on reading quality, latency and cost.

## Context

- plan.md **PHASE 11**, and **D-11.1 (REOPENED)** — the 8B pick was never measured. Its predecessor
  Qwen2.5-VL-7B scored 0.962/0.943 precision/recall on synthetic imprinted text and collapsed to
  **0.719/0.858 on real burned-in text** (arXiv 2511.02014).
- `harness/harness.py` — the existing `apply_verifier` path built in Phase 6; primary vs verifier
  timing is already split there.
- `harness/runners/base.py` — `Runner`; the verifier sets `OCROutput.box_free=True`.
- 13d's outputs — the reader accuracy table, CPU seconds/crop, and the negative-control counts for
  each candidate. **Reuse them; do not re-measure.**
- CLAUDE.md §4 — Qwen3-VL is ✅ only if self-hosted.

## Constraints

- **Keep the primary's box, swap only the string.** The verifier is box-free and cannot relocate
  text. Any code that lets it move or add a box is wrong.
- **The abstain threshold (D-11.2) is tuned to the cost asymmetry**, not borrowed from a
  detect-and-redact threshold. Destroying a valid token and leaking PHI are not equally bad; state
  which way you are trading and why, in a comment at the definition site.
- **Gate on the score family you actually have.** If the primary is PP-OCRv6, its confidence is a
  line-inherited *recognition* score — a single bad word inside a clean line will not stand out. Say
  so where the threshold is set. Never share a threshold across score families.
- **A verifier that invents text on blank controls is disqualified** regardless of accuracy elsewhere
  — check 13d's negative-control counts before wiring any candidate.
- Verifier identity is part of run identity (13a): `verifier_model_name` + `verifier_version` +
  the prompt constant.
- Do **not** modify `matching.py`, `metrics.py`, or `normalize()`.

## Steps

1. Enter **Plan Mode**. The plan must enumerate: which candidate wins on 13d's evidence and why,
   files to create/edit, the threshold's initial value and the sweep you will run to set it, and the
   comparison table's exact columns. Present D-11.1 as a decision with the 13d numbers in front of
   it — **stop and wait for my answer before coding.**
2. Implement `runners/run_qwen3vl.py` (or the chosen candidate) as a box-free `Runner` wired as
   `verifier_func`.
3. Synthetic tests: box preserved exactly, string swapped only when below threshold, abstain path,
   determinism, and no egress.
4. Write the exact commands Arnav runs for the real primary-only vs primary+verifier comparison,
   emitting PHI-free aggregates only.

## Output format

Plan + the D-11.1 recommendation first. Then the diff, test output, and the command list for Arnav.

## Done when

- `pytest` green; `ruff check .` clean.
- A test proves the verifier cannot alter a box.
- D-11.2's threshold has a written rationale tied to the cost asymmetry, not a copied number.
- The primary-only vs gated comparison is specified down to the columns, ready to run.

## Finally

Spawn one fresh-context subagent. Scope: can the verifier change a box on any path, can it run
without the primary's confidence being the gate, and is there any network egress? Correctness and
constraint violations only.
