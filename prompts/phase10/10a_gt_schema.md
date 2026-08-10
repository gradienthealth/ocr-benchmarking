Phase 10a from plan.md — the 🟢 PHI-FREE sub-step. Build `ground_truth/gt_schema.py`: the frozen
`gt.csv` column spec + validator + loader. Phase 10a ONLY — do not start 10b/10c/10d/10e and do not
create render.py, seed_tesseract.py, review_gt.py, review_ui.html, or build_gt.py.

## PHI ground rules (override everything else)
- Never Read, cat, grep, or otherwise open: any `.dcm`/`.tar`/`.png`/`.jpg`/`.tif`/`.npy`,
  `ground_truth/gt.csv`, `ground_truth/review/*.json`, any `raw_response*`, or rows of `manifest.csv`.
  This sub-step needs none of them — it is pure schema logic tested on synthetic fixtures.
- Do not weaken `.gitignore`, `.claude/settings.json`, `.claude/hooks/block_phi_read.py`, or
  `.githooks/pre-commit`. 10a needs no changes to any of them (`ground_truth/*.csv` is already
  ignored). If you think a change is required, stop and ask.
- A human will run this validator against the real `gt.csv` and may paste its output back to you, so
  every error/warning string must be PHI-free by construction: row index + column name + error code
  only — never `token_text`, never `series_uid`, never a whole row. The validator returns a report
  object; it must never `print()` PHI.

## Start here (plan.md §0 working rhythm)
Restate Phase 10a in ≤8 bullets, then list the decisions below with your recommendation for each,
then STOP and wait for my answers. No code in that first turn.

## Decisions to surface (do not guess)
1. **Packaging.** `ground_truth/` is top-level (plan.md's repo-layout tree, `.gitignore`, and
   `.claude/settings.json` all agree; it is NOT under `harness/`). It does not exist yet. Decide:
   add `ground_truth/__init__.py` and `ground_truth*` to `pyproject.toml`'s
   `[tool.setuptools.packages.find] include`, or leave it import-by-rootdir the way `tests/` works
   today. Say which and why.
2. **"Inside the image" box check.** The CSV has no width/height columns; 10b is the step that emits
   the PHI-free `(image_id, frame_idx, w, h, sha256)` render manifest. Propose the signature: an
   optional `image_sizes: Mapping[str, tuple[int, int]]` argument with the bounds check skipped *and
   reported as skipped* when absent, vs. required.
3. **Loader shape.** `harness/harness.py` takes `ground_truth: dict[str, list[GTToken]]`. Confirm
   `gt_schema` exposes a loader returning exactly that shape (so 10e and Phase 13 don't each
   reinvent it), or argue for validate-only.
4. **Label vocabulary.** `harness/metrics.py:32-33` already defines `KEEP = "KEEP"` / `PHI = "PHI"`.
   Decide: import them, or define `ALLOWED_LABELS` here plus a test asserting it equals
   `{metrics.PHI, metrics.KEEP}` so the two cannot drift.
5. **Failure mode + header strictness.** Raise on first violation vs. collect all violations into a
   report with counts (a human running this once wants the full list). And: exact ordered header
   match vs. tolerating reordering.
6. **Blank controls.** Confirmed-blank `ct_axial` frames get zero rows in `gt.csv`, so a header-only
   CSV is valid and an image with no tokens is simply absent. Confirm the validator therefore cannot
   check set membership/completeness (that is 10e's job against the manifest) and state it in a
   comment so nobody later adds a bogus check.
7. **Reader.** pandas (`harness/manifest.py` reads with `dtype=str`) vs stdlib `csv`. No new
   dependency either way.

## Spec
- Columns, in this exact order: `image_id, series_uid, modality, vendor, stratum, frame_idx,
  token_text, x0, y0, x1, y1, label` — the `GTToken` fields at `harness/contract.py:96-118` with
  `bbox` flattened to four columns. D-2.3: the dataclass stays in `contract.py`; the CSV spec and
  validation live only in `gt_schema.py`.
- **Anti-drift test is mandatory:** introspect `dataclasses.fields(GTToken)` and assert the column
  spec is derivable from it, so adding a field to `GTToken` fails this test instead of silently
  producing a stale CSV spec.
- bbox convention (D-2.2, `harness/contract.py:100-104`): axis-aligned, TOP-LEFT origin, y increases
  downward, pixels of the fed image. Validate `x0 < x1`, `y0 < y1`, non-negative, and within `(w, h)`
  when sizes are supplied.
- `label` ∈ exactly `{"PHI", "KEEP"}` — binary, case-sensitive. There is no `OTHER`; do not accept it
  or describe it as valid.
- `frame_idx` a non-negative integer; coords numeric; `token_text` stored raw — no stripping, no case
  folding, and no `normalize()` anywhere in this module. `normalize()` is frozen at
  `harness/contract.py:121-139` and applied at match time only; normalizing here would
  double-normalize the yardstick.
- Do not modify `contract.py`, `matching.py`, `metrics.py`, `harness.py`, `aggregate.py`, `report.py`,
  or any runner.
- Style: Python 3.12, `from __future__ import annotations`, full type hints, ruff-clean at
  line-length 100. Log every resolved decision as a code comment/docstring in the module.

## Tests — synthetic only
- One new file `tests/test_gt_schema.py`, following `tests/test_contract.py` conventions (module
  docstring stating every token is fabricated and no PHI appears).
- Build GT rows from `tests/synthetic.py` — `make_scene(seed=0)`, `make_synthetic_image`,
  `make_blank_image`, fake tokens like `CMFN`/`GRDN1234`/`ACC-0001`. Write every CSV under pytest's
  `tmp_path` — never under `ground_truth/`, never a real path.
- Cover at minimum: round-trip (write scene GT → validate clean → load → equals the original
  `GTToken` objects, `token_text` byte-identical); header drift vs. `dataclasses.fields(GTToken)`;
  rejected labels (`"OTHER"`, `"phi"`, `""`); `x0 >= x1`; `y0 >= y1`; negative coords; out-of-bounds
  when sizes are supplied; non-integer `frame_idx`; missing column; extra column; header-only file is
  valid (blank control); missing file; and a **PHI-leak test** asserting no string in the returned
  report contains any fixture `token_text` or `series_uid`.
- No skips, no `xfail`, no network, no real data.

## Verify
```
python -m pytest tests/test_gt_schema.py -q
python -m pytest -q
ruff check .
```
Report the full suite's pass/skip counts verbatim. Skipped > 0 means not done — say so loudly rather
than calling it green.

## Then
1. Show me the diff, then commit exactly the two files in one commit naming the phase (e.g.
   `Phase 10a: frozen gt.csv column spec + validator (gt_schema.py)`). Do NOT tick README's Phase 10
   checkbox — 10b–10e remain. Do NOT push (`git push` is hook-denied).
2. Report: the module and test file, the verbatim pytest summary line, and a ≤10-line log of the
   decisions as resolved. No prose recap of the code.
