Phase 10e from plan.md — build `ground_truth/build_gt.py`.

## 0. PHI RULE — read this before anything else
🔴 This phase touches PHI. **You write the code; I run it; you read only the PHI-free summary.**

**NEVER read, print, decode or echo:** `ground_truth/gt.csv` · anything under `ground_truth/review*/`
or `ground_truth/seed*/` · any `.dcm` / `.png` / `.npy` / render · any `raw_response` ·
`ground_truth/render_backmap*` · `ground_truth/render_inputs*` · any row of `manifest.csv` or
`gt_sample_v2.csv`. No exceptions, no "just to check the shape." Write code that consumes these
files; never open them yourself.

Do not run `build_gt.py` against real data — I run it. Never weaken a PHI rail (see decision 9).

## 1. Goal
`ground_truth/build_gt.py` turns the Phase-10d human review JSONs into the frozen, hashed
`ground_truth/gt.csv` (gitignored) and writes three committed artifacts beside it:
`gt.csv.sha256`, `gt_set.sha256`, `text_presence_v2.csv`. Plus `tests/test_build_gt.py`,
synthetic fixtures only.

## 2. Context — read these first (source + docs only)
- **plan.md**: Phase 10 header gate · the 10d "Data in and out" block · the 10e sub-step · the Phase
  10 decisions block · START HERE (966 series / 66 `ct_axial`, canonical manifest) · PHASE 5 metric 4
  · the Phase 13 step that verifies `gt.csv`'s sha256 and hard-aborts on mismatch (D-13.1) · §3
  cross-cutting checklists.
- **CLAUDE.md** §0, §6 rules #1 / #3 / #7 / #8, §7.
- `harness/contract.py` — `GTToken` (line 96), `normalize()` (line 121).
- `harness/manifest.py` — whole file, it is short. `.gitignore`.
  `.claude/hooks/block_phi_read.py` — note `PHI_NAME` (line 27).
- **10a–10d are built and committed; every interface 10e consumes is already pinned in code.** Read
  and never modify: `ground_truth/gt_schema.py` (10a — the frozen column spec + validator you
  import) · `ground_truth/render.py` (10b) · `ground_truth/seed_tesseract.py` (10c) ·
  `ground_truth/review_gt.py` + `ground_truth/review_ui.html` (10d — the review-record writer, i.e.
  the exact field names and states 10e reads).
  *(Those four source files were briefly unreadable: two `permissions.deny` globs, `seed_*/**` and
  `review_*/**`, matched the `.py`/`.html` sources as well as the PHI dirs. Fixed 2026-08-09 by
  naming the PHI dirs explicitly and narrowing the wildcards to `*.json`. Do not re-widen them.)*

## 3. State of the data — measured 2026-08-09, PHI-free aggregates
**These are CONTEXT, not pass conditions.** Count everything from the files at run time and print the
counts, so a changed input is visible at a glance instead of silently scoring the wrong set.

- Scored set = **199 images**: the id set of `renders/v2/render_manifest.csv` == `seed_v2/` ==
  `ground_truth/review/`. Drawn set was **202** (`gt_sample_v2.csv`); 3 series never rendered
  (`us_philips` 1, `us_siemens` 1, `mg_tomo` 1 — scattered, so not a systematic render bug).
- Review records: **0 deferred**, 136 `edited`, 63 `accepted`, all `round: 1`.
- **97 of 199 records hold zero tokens** → `gt.csv` gets rows for **102** images. Blank breakdown:
  `ct_axial` 64 · `mg_tomo` 16/16 · `ct_scout` 12/12 · `mg_2d` 4 · `ct_secondary_capture` 1.
- **2 `ct_axial` images have text** → they leave the control set; **blank-control survivors = 64**.
- `manifest.csv` is now the canonical **966 rows / 66 `ct_axial`** pull, not the stale sample.
- `middle_frame_index == number_of_frames // 2` on all 966 rows — it encodes no human judgement. It
  disagrees with the rendered frame on **9 of 199** images (all ultrasound) because
  `render_inputs_v2.csv` left `frame_idx` blank on all 202 rows, so `render.py` fell back to its own
  `n // 2` over the frames actually present in the tar. For those 9 the tar holds far fewer frames
  than the `NumberOfFrames` tag claims.

## 4. Decisions — ALL RESOLVED 2026-08-09
Implement them. Do not re-open, re-negotiate, or invent past them. Log each as a short comment where
the code governs it.

1. **D-10.7 → self-agreement on a ~10% slice.** ~20 of the 199 get re-reviewed after a gap into
   `ground_truth/review_r2/`; 10e reports token-level round-1-vs-round-2 agreement over the ids
   present in both. **Round 2 has not been run yet.** If `review_r2/` is absent or empty, the summary
   prints `self-agreement: not yet run (round 2 absent)` and **no number** — it must be impossible for
   this summary to print an agreement figure it cannot compute. Round 2 never overwrites round 1 and
   never feeds `gt.csv`; round 1 is the GT.
2. **D-10.2 → full, untruncated, lowercase-hex sha256** over `gt.csv`'s bytes. The 8-char truncation
   is 10b's `image_id` scheme and is unrelated — do not truncate this one.
3. **10a–10d are committed.** Import 10a's validator; never reimplement it. Take the review-record
   shape from 10d's committed `review_gt.py` — read the code for the exact field names and states
   rather than assuming them. Do NOT build or modify 10d.
4. **D-10.8 → option A, a narrow keyed accessor in `harness/manifest.py`.** The join is two hops:
   `image_id → series_uid` from `ground_truth/render_backmap_v2.csv` (**that** is the back-map for
   this set — there are five, do not guess), then `series_uid → vendor / stratum / modality` from
   `manifest.csv` **through the adapter**. Add `series_uid` to `_COLUMN_MAP` as an **index key only**
   and add one accessor returning exactly `(vendor, stratum, modality)` per uid. No accessor may
   return an identifier — that invariant in `manifest.py`'s docstring must survive. Do not read
   `manifest.csv` anywhere else, do not widen the whitelist further, and do not re-derive any of it
   from filenames or model strings (CLAUDE.md rule #7). Amend the D-10.8 line in plan.md in the same
   commit so the record matches the code.
5. **Scored set → `renders/v2/render_manifest.csv` defines it** (PHI-free, 10b-authored, 199 rows).
   Two failure classes, handled differently:
   - **Review completeness → hard failure.** Any image in the scored set with no review record, or
     still `deferred`, raises and exits non-zero with **no `gt.csv` written**. Check
     **bidirectionally** — an orphan review record whose `image_id` is absent from the render manifest
     also fails: it means a pilot-round file bled in, or the `image_id` formula changed and produced a
     disjoint set.
   - **Drawn-vs-rendered → recorded, never fatal.** Read the drawn count from `gt_sample_v2.csv` and
     print `drawn / rendered / reviewed / deferred / missing`. The 3 unrendered series are a data fact
     about finished work; aborting on them would only invite a `--force` flag. **No `--force`, no
     `--skip-missing`, ever.**
6. **`frame_idx` comes from `renders/v2/render_manifest.csv`** — the frame actually rendered and
   annotated, which is also the preimage of that image's `image_id`. `vendor`, `stratum` and
   `modality` still come from `manifest.csv` via the adapter. Print the manifest-vs-rendered
   disagreement count (expect 9) rather than reconciling it silently. Amend CLAUDE.md rule #7 to say
   `frame_idx` comes from the render manifest, and why: the rule exists to forbid re-deriving from
   filenames and model strings, and the render manifest is a recorded fact, not a re-derivation.
7. **Blank images → emit `ground_truth/text_presence_v2.csv`** (`image_id,has_text`; PHI-free,
   committed, same pattern as the existing `text_presence_*` files). A blank image is ABSENT from
   `gt.csv`, so from inside the CSV "reviewed and found blank" and "never reviewed" are
   indistinguishable — this file is the only place that distinction survives, and 97 of 199 images
   depend on it. The summary must also emit a **named warning line** for any stratum whose
   `images_with_text == 0` (expect `mg_tomo` and `ct_scout`), so a downstream recall denominator of
   zero cannot silently read as a saturated 100%.
8. **Set hash → also write `ground_truth/gt_set.sha256`**, the sha256 over the sorted,
   newline-joined `image_id`s of the scored set. `gt.csv.sha256` pins *contents*; this pins *scope*,
   and the two are independent: because blanks are absent by construction, a run over all 199 images
   and a run over only the 102 text-bearing ones produce a byte-identical `gt.csv` and the same
   content hash. Both hashes go in the summary; both get committed.
9. **PHI rails are verified present — do not re-widen.** `ground_truth/review/` is covered in all
   three places (`.gitignore`, the hook's `PHI_NAME`, `permissions.deny`). Adding a deny pattern is
   strengthening a rail and is in scope; removing or loosening anything in that hook is not, ever.

## 5. Constraints (non-negotiable)
- **Input is the 10d review JSONs, never the raw 10c Tesseract seed. That is the gate** — no seed
  fallback, no "seed if no review" path, no `--use-seed` flag.
- **Columns:** `image_id, series_uid, modality, vendor, stratum, frame_idx, token_text, x0, y0, x1,
  y1, label` — `GTToken` field for field with `bbox` flattened (top-left origin, pixels). `label` is
  binary, exactly `PHI` or `KEEP`. Validate every row against the 10a schema/validator.
- **Store `token_text` RAW** — no strip, no case folding, no cleanup, no Unicode fiddling.
  `normalize()` (`harness/contract.py:121`) is frozen and applied at match time; normalizing here
  double-normalizes and corrupts the yardstick. Do not import or call `normalize()` in this script.
- **Do not touch** `harness/contract.py`, `normalize()`, `matching.py`, `metrics.py`, `aggregate.py`,
  or any runner. The only permitted edit outside 10e is decision 4's keyed accessor in
  `harness/manifest.py`.
- **Blank controls:** `stratum == "ct_axial"` gets **ZERO rows** — that zero is the hallucination
  floor. A "blank" frame that turns out to have text gets normal rows and **leaves** the control set;
  compute and record the survivor count. **Never hard-code 64, 66, 102, 199, 202 or 966 as a pass
  condition** — count from the loaded files, and print the loaded manifest's row count and `ct_axial`
  count in the summary.
- **Freeze + hash:** `gt.csv` (gitignored) plus `gt.csv.sha256`, `gt_set.sha256` and
  `text_presence_v2.csv` (committed). Phase 13 re-verifies the content hash and hard-aborts on
  mismatch (D-13.1). Any later change to `gt.csv` invalidates every number already reported (CLAUDE.md
  rule #8) — if the script is re-run over an existing `gt.csv`/`.sha256` and **either** hash changes,
  say so **loudly** in the summary as an invalidation warning. Never overwrite quietly.
- **Nothing PHI to stdout, ever.** No token strings, no unhashed filenames, no box contents, no
  DataFrame previews, no traceback that renders a row. The summary is the only output surface and
  carries exactly: total row count; images with rows vs blank; per-stratum, per-label and per-vendor
  counts; blank-control survivor count; both hashes; the loaded-manifest counts; the
  drawn / rendered / reviewed / deferred reconciliation; the frame-index disagreement count; the
  empty-stratum warnings; the 10d review + seed-quality stats; and the self-agreement line (or its
  "not yet run" form). Write it to a file **and** print it — both PHI-free by construction.
- **Determinism:** sorted inputs, stable column order, explicit CSV quoting/newline/encoding, so a
  rerun over unchanged inputs is byte-identical and both hashes are stable. No ordering *requirement*
  on the scored set (D-1.1 resolved — reproducibility is a fixed + hashed set, not an order); do not
  reintroduce one.
- **No new dependencies** — stdlib + existing pins (pandas, hashlib) only.

## 6. Steps
1. **Plan Mode first.** This touches five files. Produce a plan that enumerates: every file to edit,
   the specific functions/symbols added or changed in each, the order of operations, and which
   decision above governs each piece. Then implement.
2. Restate the plan in ≤8 bullets, then write `ground_truth/build_gt.py`.
3. Add decision 4's accessor to `harness/manifest.py`, and the plan.md + CLAUDE.md amendments from
   decisions 4 and 6.
4. Write `tests/test_build_gt.py` (§7) and get it green.
5. **Fresh-context reviewer subagent.** Spawn one subagent with no prior context to review the diff
   only — it must not open any file on the never-read list. Constrain it to flag **only** correctness
   bugs, PHI-safety violations, or departures from the decisions and constraints above; everything
   else is optional and should be reported as such. Fix what it confirms.
6. Commit (§9). Do not push.

## 7. Tests (`tests/test_build_gt.py`)
Synthetic only — fake review JSONs, fake tokens (`CMFN-00421`-style), fake manifest rows, fake render
manifest, `tmp_path`. Reuse `tests/synthetic.py` / `tests/conftest.py` where it fits. Zero real data,
zero real images, **zero skips — a skipped test is a failed test here.** At minimum:
- a missing review record fails the build (raises / non-zero, and no `gt.csv` is written);
- a `deferred` review record fails the build the same way;
- an orphan review record not in the render manifest fails the build (the bidirectional half);
- a drawn-but-unrendered series is reported in the reconciliation line and does **not** fail the build;
- `ct_axial` images contribute zero rows, and the survivor count is right when one "blank" image turns
  out to have text;
- `token_text` round-trips byte-identical, including a case-varied and a whitespace-bearing token;
- a `label` outside `{PHI, KEEP}` is rejected;
- `gt.csv.sha256` matches a recomputed hash of the written `gt.csv`, and a rerun over unchanged inputs
  reproduces both hashes;
- `gt_set.sha256` changes when a blank image is added to the scored set even though `gt.csv` stays
  byte-identical (the case that motivates the second hash);
- `frame_idx` comes from the render manifest, not `manifest.csv`, when the two disagree;
- `vendor`/`stratum`/`modality` come from the manifest adapter, not from the filename or the review
  JSON;
- `text_presence_v2.csv` lists every scored image with the right `has_text`, and a stratum with zero
  text-bearing images produces the named warning line;
- with no `review_r2/` present the summary prints the "not yet run" form and no agreement number; with
  a synthetic `review_r2/` present it prints a computed one.

Verify with `python -m pytest tests/test_build_gt.py -v`, then `python -m pytest -q`, then
`python -m ruff check .`

## 8. Scope
Phase 10e only, plus decision 4's accessor and the plan.md / CLAUDE.md amendments from decisions 4
and 6. `review_gt.py` / `review_ui.html` (10d) are committed — read them, never modify or rebuild
them. Do not touch Phase 11+ (verifier arm, real runs, reports). Do not run anything that reads real
PHI.

## 9. Done when
- `python -m pytest -q` green with nothing skipped; `ruff` clean.
- Every decision logged as a short comment where the code governs it: why raw `token_text`, why the
  gate raises, why the manifest is the source of `vendor`/`stratum`, why the render manifest is the
  source of `frame_idx`, and what D-10.7, D-10.8, the scored set, the set hash and
  `text_presence_v2.csv` resolved to.
- One commit naming the phase (e.g. `Phase 10e: review JSONs → frozen, hashed gt.csv`).
  **Staged:** `ground_truth/build_gt.py`, `tests/test_build_gt.py`, the `harness/manifest.py`
  accessor, the plan.md + CLAUDE.md amendments.
  **NOT staged:** `gt.csv`, anything under `ground_truth/review*/` or `seed*/`, any render, any
  back-map. `gt.csv.sha256`, `gt_set.sha256` and `text_presence_v2.csv` are produced by **my** run,
  not yours — they get staged after I run it.
  Check `git status` explicitly before committing. **Do not push.**

## 10. Report
What was built · what was verified by which command · what was skipped or left open · any PHI-safety
flag. Surface partial work; never hide it behind "done."
