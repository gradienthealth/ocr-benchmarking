# 13a — D-13.5: config identity 🟢 PHI-free

> **Run this FIRST.** Every other Phase-13 prompt (13b–13g) adds an arm, and until a run can
> identify its own config, arms silently average together. Nothing downstream is trustworthy
> without this.

---

## Goal

`aggregate()` can distinguish two runs of the *same library at the same version* that used
different configs, and refuses to blend them — so docTR-stock, docTR-tuned and docTR-parseq
produce three separate rows instead of one meaningless average.

## Context

- `harness/aggregate.py` — the mixed-batch guard keys on
  `(model_name, version, verifier_model_name, verifier_version)`. That tuple is the bug: docTR
  stock and docTR with `reco_arch='parseq'` share both `model_name` and `doctr.__version__`.
- `harness/runners/base.py` — `Runner.model_name` / `.version` / `.version_source`; read the
  `version_source` docstring before touching anything, it explains why versions are never
  hand-typed.
- `harness/harness.py` — `run_harness()` and its `run_metadata`.
- `harness/report.py` — whatever labels a run in the output tables must carry the new identity too.
- plan.md, PHASE 13, "Trap 2" — the original statement of this problem.
- Arms that will exist within the week: docTR stock, docTR tuned, docTR+parseq, PP-OCRv6 stock,
  PP-OCRv6 tuned, EasyOCR, SVTRv2-reader, Qwen3-VL-4B-reader. Design for ~10.

## Constraints

- **Do NOT touch** `matching.py`, `metrics.py`, or the frozen `normalize()`. This is an identity
  and aggregation change only — the scoring path stays fixed so the comparison stays fair.
- **Do NOT weaken the mixed-batch guard.** It must still refuse to concatenate runs that differ;
  it is gaining a dimension, not losing strictness.
- The identity must be **derived from real config**, never hand-typed — same rule as `version`
  (CLAUDE.md rule #9). A runner that forgets to declare it must fail loudly, not default to
  something that collides.
- No PHI: work on synthetic fixtures only (`tests/synthetic.py`).

## Steps

1. Enter **Plan Mode**. The plan must enumerate: every file to edit, the exact
   functions/attributes to add or change, the order of edits, and which existing tests will need
   updating.
2. In the plan, present the identity design as an explicit decision with 2–3 options and a
   recommendation — e.g. an explicit `config_id` attribute on `Runner` vs. a structured
   `variant` suffix folded into `model_name` vs. a hash over a declared config dict. State the
   trade-off for each in one line: readability in report tables, collision risk, and how loudly
   a runner that forgets to set it fails. **Stop and wait for my answer before writing code.**
3. Implement the chosen design across runners, harness, aggregate and report.
4. Add tests: two runs identical except for config **must** aggregate separately; two genuinely
   identical runs must still aggregate together; a runner that declares no identity must raise.
5. Update the existing three runners to declare their identity.

## Output format

Plan first (no code). After I approve: the diff, then the test output.

## Done when

- `pytest` passes with the new tests, and the pre-existing suite is unchanged in count except for
  the additions (report any test you had to modify and why).
- `ruff check .` is clean.
- A one-paragraph note I can paste into plan.md recording D-13.5 as RESOLVED, naming the chosen
  design and the rejected alternatives.

## Finally

Spawn one fresh-context subagent to review the diff. Scope it to: does any code path still let two
different configs collide into one aggregate row, and does any runner silently pass without
declaring identity? Flag only correctness bugs and violations of the constraints above — treat
style and refactor opportunities as out of scope.
