## ⭐ STATE — updated 2026-08-05. READ THIS FIRST; the prompt below is the original brief.

**Done and committed** (`bd37527`, one commit, not pushed): `ground_truth/render.py` +
`tests/test_render.py` (49 tests, 0 skipped, ruff clean, full suite 291 passed / 1 pre-existing
xfail). Implemented: D-10.2 (8-char `image_id`, collision gate raising before any write, `--id-len`
recovery), forced RGB, D-10.8 (explicit `series_uid,path,frame_idx` input list; `manifest.csv` never
read), frame selection + `fallback_used` provenance, the two-file PHI split, and the add-only rail
extensions for `render_backmap`/`render_inputs`.

**BLOCKED — D-10.1 is still open, so the renderer cannot produce a single pixel.**
`voi_to_uint8()` raises `WindowPolicyUnsettled` on purpose: no guessed windowing policy may get
baked into `gt.csv`.

### RUN LOG — what I have already run (so I don't redo it, and don't assume more than happened)

| When | Command | Outcome |
| --- | --- | --- |
| 2026-08-05 ~20:38–20:51 | `python scripts/tag_census.py --limit-per-stratum 40` | **INCOMPLETE — produced no usable output.** Ran with the PRE-fix script, which printed a real SOPInstanceUID to stderr (that is what the `_silence_pydicom()` patch fixes). Progress reached `20/40` within a stratum. `census_out/` is **empty**: the script only writes `tag_census.md` after every stratum finishes, so an interrupted run leaves nothing behind. |

**Therefore D-10.1 is still fully blocked and the census must be run again.** Do not proceed as if
the table exists — verify with `ls census_out/tag_census.md` before assuming anything.

**Leftover to clean up:** `census_tars/` still holds a ~125 MB series `.tar` (raw PHI on local disk).
The script unlinks each tar in a `finally`, so its presence means that run was interrupted. If the
script is no longer running, delete it — Claude will not, both because it is destructive and because
the file may still be in use. `census_tars/` is gitignored (that rule was broken by a trailing inline
comment until 2026-08-05 and is now fixed), so it was never at risk of being committed.

### My next actions, in order
1. Re-run the census (the script now silences pydicom, so it can no longer print a UID):
   `python scripts/tag_census.py --limit-per-stratum 40`
   It downloads a tar per sampled series, so budget the time; confirm `census_out/tag_census.md`
   exists and ends with a full per-stratum table before moving on.
2. Paste **only the contents of `census_out/tag_census.md`** into a new session — the file, never
   terminal output. A pasted stderr line leaked a real SOPInstanceUID on 2026-08-05; that is what
   the silencing fix addresses.
3. Settle D-10.1a/b/c from that table, plus these six the census must also answer:
   (a) any stratum using `VOILUTFunction = SIGMOID` — honour the vendor curve or force `LINEAR` for
   cross-engine comparability? (b) when Window Center/Width is multi-valued: index 0, a named window
   via `WindowCenterWidthExplanation`, or the widest — and does the choice need recording per image
   (there is no column for it today)? (c) any `PixelRepresentation = 1` (signed) or `BitsStored > 8`,
   which decides whether the modality LUT is load-bearing or a no-op? (d) for "neither tag present"
   with `BitsStored == 8` (secondary capture / US): pass-through or min/max stretch — a stretch
   alters already-display-ready pixels? (e) any `PALETTE COLOR` (currently a loud failure, not a
   render)? (f) how often the manifest frame index falls out of range per stratum — a high rate means
   the index's axis semantics are wrong before 966 series get rendered.
4. Claude then implements `voi_to_uint8` + its precedence tests and **amends** `bd37527`, so Phase
   10b stays one commit.

### Open items that are mine, not Claude's
- Sync `/etc/claude-code/hooks/phi_guard.py` (root-owned) with the new `render_backmap` /
  `render_inputs` patterns now in `.claude/hooks/block_phi_read.py`.
- Decide whether to widen the hook's **Bash** patterns: the new rules gate the `Read` tool only, so
  `cat`/`grep` on a UID-bearing CSV is still allowed — a pre-existing gap that also applies to
  `gt.csv`, deliberately left out of the 10b commit.
- `tests/test_harness.py` has 2 ruff errors (`I001`, `F401`) that pre-date this work.
- Carry-over lessons from this session (library warnings as a PHI egress path, non-conformant UIDs,
  assert-on-the-artifact, git's lack of inline comments, atomic writes) are written into
  `prompts/phase10/10d_review_ui.md` under "Carry-overs from the 10b / tag-census session".

---

Phase 10b from plan.md — build `ground_truth/render.py`: DICOM → deterministic PNG of the middle
frame. 🔴 PHI-TOUCHING: you write the script, **I run it**, and you never read its output.

## PHI rules (these override everything else in this prompt)
- NEVER `Read`, render, decode, print, or otherwise pull into context: any `.dcm`/`.dicom`/`.tar`,
  any `.png`/`.jpg`/`.tif`/`.npy`, any crop, `ground_truth/gt.csv`, `ground_truth/review/*.json`, or
  any engine `raw_response`. Not even "just to check it looks right" — a human does that
  (CLAUDE.md §0, §6.1).
- Do NOT run `render.py` yourself, and do not run anything that downloads from
  `gs://gradient-central-intern-ocr-data`. Images live only in GCS (`series/<series_uid>.tar` →
  `instances/<sop_instance_uid>.dcm`) and the sandbox blocks `googleapis` egress on purpose. I pull
  the data and I run the script.
- All rendering is local (pydicom + Pillow + numpy). Nothing to any third party — no cloud OCR, no
  API, no upload, no `requests`.
- Do not weaken any rail: `.gitignore`, `.claude/hooks/block_phi_read.py`, `.claude/settings.json`
  deny rules stay as strict or stricter.
- You may run: `pytest` on synthetic tests, `ruff`, `git` (no push).

## Goal
One new file `ground_truth/render.py` + one new test file `tests/test_render.py`, such that: given a
set of local DICOM instances, it writes deterministic PNGs of the **middle** frame into a gitignored
directory and emits a **PHI-free** manifest of `(image_id, frame_idx, w, h, sha256)` — the only
artifact I will ever show you. Byte-identical output on rerun.

## Context — read these, nothing else
- plan.md: the Phase 10 header gate, the 10b sub-step, the Phase 10 decisions block (D-10.1, D-10.2,
  D-10.8), the acceptance criteria and review checklist, §0 operating principles + working rhythm,
  §3 cross-cutting checklists, and START HERE (the local `manifest.csv` is the stale 1000-row sample;
  canonical is 966 series / 66 `ct_axial`).
- CLAUDE.md §0 (golden rule), §6 (hard stops — esp. rule #7 vendor/stratum/frame_idx come from the
  manifest adapter, and rule #5 never call this data de-identified), §7, §8 (`BurnedInAnnotation` is
  unreliable; frame 0 is a banner/title screen).
- `harness/manifest.py` — the one audited door onto `manifest.csv`. Read `_COLUMN_MAP` and the
  `ManifestView` accessors before assuming you can use it (see D-10.8 below).
- `harness/harness.py` — the `ImageRef` dataclass (`id, path, w, h, stratum, modality, vendor,
  frame_idx`). Your render manifest should line up with what a later phase needs to build `ImageRef`s,
  but do NOT edit `harness.py`.
- `scripts/preview_ct_axial.py`, `scripts/scan_text_poscontrol.py` — existing human-run ad-hoc
  renderers. Learn their shape (tar streaming, `stop_before_pixels` header pass, VOI LUT,
  MONOCHROME1 invert, delete the tar after use, counts-only stdout). They are provisional, **not**
  the pinned encoder — do not just copy them.
- `.gitignore` (`renders/`, `*.png`, `ground_truth/*.csv`, `*.log` already blocked), `pyproject.toml`.

Environment facts — build on these, don't re-derive:
- Installed and pinned: `pydicom==3.0.1`, `Pillow==11.1.0`, `numpy==2.2.1`; decoder plugins
  `pylibjpeg` and `gdcm` are present, so compressed transfer syntaxes will decode. Add **no** new
  dependency without asking me.
- Use the modern `pydicom.pixels` API (`apply_voi_lut`, `apply_modality_lut`, `convert_color_space`),
  not the deprecated `pydicom.pixel_data_handlers.util` path the old scripts import.
- `ground_truth/` is **top-level**, not under `harness/` — `.gitignore` and
  `.claude/settings.json` deny rules are written to that path.

## Decisions — SETTLED 2026-08-05. Implement these; do not re-litigate them.

- **D-10.2 — RESOLVED: `image_id = sha256(f"{SOPInstanceUID}|{frame_idx}".encode("utf-8")).hexdigest()[:8]`.**
  Exactly that preimage: UTF-8, a literal `|` separator (so `UID1` + `23` cannot collide with `UID12`
  + `3`), `frame_idx` formatted as a plain decimal int with no padding. Truncated to **8 hex chars**
  for readable filenames, which makes collisions unlikely but not impossible — so a collision must be
  **detected and raised**, never silently allowed to merge two images' ground truth. The check runs
  across the whole render set *and* against any previously-rendered ids in the back-map, before
  anything is written. Recovery: **re-running unchanged cannot help** — sha256 is deterministic and
  reproduces the same collision — so the id length must be raised instead. Expose `--id-len`
  (default 8) and make the error say so, naming both colliding ids and the fix, plus the warning that
  changing `--id-len` changes **every** `image_id`, so it is only safe before `gt.csv` exists (after
  that it is a rule #8 invalidation of every reported number).
- **PNG mode — RESOLVED: force `RGB` for every image**, monochrome included (replicate the single
  channel to three). One uniform input type for every engine, runner, and test fixture; nothing
  branches on channel count. Apply the `MONOCHROME1` inversion *before* the channel expansion.
- **D-10.8 — RESOLVED: `render.py` never touches `manifest.csv`.** It takes an explicit input list of
  local `.dcm`/tar paths + frame indices that **I** generate and pass in. Do **not** widen
  `harness/manifest.py`'s `_COLUMN_MAP`, do not add a per-row accessor, do not read the manifest
  here. The `series_uid → vendor/stratum/modality` join is 10e's problem, built once where it is
  actually needed. Rendering needs no vendor/stratum at all.
- **Frame selection — RESOLVED: manifest index first, `n // 2` fallback, and the render is the
  single source of truth for `frame_idx`.** Use the frame index I pass in when it is in range; when
  it is out of range, fall back to `n_instances // 2` (or `n_frames // 2` for a multi-frame
  instance). This cannot desync GT from what the engines read — there is exactly one PNG per image,
  and both the annotator and every engine read that same file. The only risk is a *provenance* lie,
  so:
  - the PHI-free render manifest records the frame **actually rendered**, not the requested one;
  - it also carries a per-image `fallback_used` flag and a total fallback count, so substitution is
    visible rather than silent;
  - 10e and `ImageRef` must take `frame_idx` from the render manifest, **never** from `manifest.csv`
    — say so in a comment so nobody re-joins it later;
  - **pin the instance order**: sort by `InstanceNumber`, tie-broken by `SOPInstanceUID`. "Index *i*
    of the series" is undefined without a fixed order, and an unpinned sort can select a different
    slice on a later run.
- **Back-map — RESOLVED: two output files, one readable by you and one never.** The PHI-free manifest
  (`image_id, frame_idx, w, h, sha256, fallback_used`) is the only thing I show you. The
  `image_id → SOPInstanceUID / series_uid` back-map is PHI-adjacent and goes in a **separate**
  gitignored, hook-denied local file keyed by `image_id` that you never read — the same split as
  `gt.csv` vs `gt.csv.sha256`. Add it to `.gitignore` and to the hook deny patterns in this commit
  (add-only; never loosen an existing rule).

## The one OPEN decision: D-10.1 (windowing) — gated on the tag census

Do **not** pick a windowing policy and do **not** write the pixel path until I give you the answers.
The census script already exists: I run `python scripts/tag_census.py --limit-per-stratum 40`, which
emits `census_out/tag_census.md` — a PHI-free table of per-stratum counts (VOI LUT Sequence present,
Window Center/Width present, both, neither, multi-valued window, `VOILUTFunction`, Rescale
Slope/Intercept, `PhotometricInterpretation`, `SamplesPerPixel`, `BitsStored`, multi-frame, and how
often the manifest frame index falls out of range). I will paste that table to you.

From it, we settle together, and you then implement exactly what I choose:
- **D-10.1a** — when both a VOI LUT Sequence and Window Center/Width are present: LUT wins
  (`prefer_lut=True`, pydicom's default and the vendor's intended display mapping) or windowing wins
  (`prefer_lut=False`, a simple inspectable linear map). Also: which value to use when Window
  Center/Width is multi-valued.
- **D-10.1b** — whether `apply_modality_lut` (Rescale Slope/Intercept) runs **before** windowing
  (pydicom's docs say it must when the dataset requires it), and how the windowed result is mapped to
  8 bits: a fixed window-range → 0-255 map with clipping, or a per-image min/max rescale. Note for
  this dataset specifically: burned-in overlay text usually sits at the pixel extremes, so a
  per-image min/max rescale is anchored by the text itself.
- **D-10.1c** — the fallback when neither VOI LUT nor Window Center/Width is present (expected to
  matter most for the `us_*` strata).

When you restate the plan (≤8 bullets), list any *further* pixel-path questions the census raises —
but do not reopen the settled decisions above.

## Constraints
- **Middle frame only, never frame 0** (frame 0 is typically a banner/title screen, so scoring it
  measures the wrong thing). Handle both shapes explicitly: a single multi-frame instance
  (`NumberOfFrames > 1`) and a multi-instance series (one `.dcm` per slice). Single-frame dose/protocol
  reports are the legitimate exception (`frame_idx = 0` because there is only one frame) — make that
  path explicit, not accidental. Selection policy and the fallback are settled above.
- **Pixel handling:** invert `MONOCHROME1`; convert `YBR_*` → RGB; apply the D-10.1 windowing policy
  once I've settled it from the census; then the agreed 8-bit mapping; then emit **`RGB` always**
  (replicate mono to three channels last).
- **Determinism is a correctness property:** sorted inputs, pinned instance ordering
  (`InstanceNumber`, then `SOPInstanceUID`), pinned Pillow save parameters (explicit `format`, fixed
  `compress_level`, fixed `optimize`, no metadata/text chunks, no timestamps), no unseeded randomness,
  no wall-clock or dict-iteration dependence. Same inputs → byte-identical PNGs and identical sha256s
  across reruns.
- **Output contract — three artifacts, exactly one of which I may show you:**
  1. the PNGs, into a gitignored directory (e.g. `renders/gt/`);
  2. the **PHI-free** manifest: exactly `image_id, frame_idx, w, h, sha256, fallback_used` — the only
     thing you ever read;
  3. the gitignored, hook-denied `image_id → SOPInstanceUID/series_uid` back-map — which you never
     read.
- **PHI never to stdout.** stdout gets counts/progress and error *types* only — never a token string,
  a DICOM tag value, a SOP/series UID, a path containing a UID, or a traceback that could render tag
  data. Wrap per-instance failures so one bad file doesn't kill the run and report them as
  `(count, error_type)` aggregates. Do not write a `*.log` containing PHI.
- **Fail loud** on anything ambiguous (missing `PhotometricInterpretation`, unexpected sample count,
  frame index out of range) rather than silently defaulting.
- Never assert or imply this data is "de-identified" or "anonymous" in code, comments, or the commit
  message — it is pseudonymized PHI (CLAUDE.md §6.5).
- Do NOT edit `harness/*.py`, the hooks, or `manifest.csv`. The only exception is an adapter change
  if I pick D-10.8 option (a).

## Tests — synthetic only, none skipped
- `tests/test_render.py`, built on **in-memory synthesized `pydicom.Dataset` objects** (`FileDataset`
  + fake numpy pixel arrays) written to `tmp_path`. Zero real `.dcm`, zero GCS, zero PHI. Follow the
  obviously-fake-values style of `tests/synthetic.py`.
- Cover at minimum: middle-frame selection for multi-frame and for a multi-instance series; the
  single-frame exception; the out-of-range fallback (and that `fallback_used` + the recorded
  `frame_idx` reflect what was actually rendered, not what was requested); pinned instance ordering
  including the `SOPInstanceUID` tie-break; `MONOCHROME1` inversion; `MONOCHROME2` left alone;
  RGB/YBR conversion; every output being mode `RGB`; the chosen D-10.1 precedence including the
  neither-present fallback; `image_id` stability for a fixed (UID, frame_idx) and the exact preimage
  format; **an 8-char collision raising** with both ids named and `--id-len` in the message;
  `--id-len 12` changing every id; **byte-identical rerun** (render twice, compare file bytes and
  sha256); manifest row shape and values; and that the PHI-free manifest contains no UID and no tag
  value while the back-map is written to a separate gitignored path.
- Verify with `.venv/bin/python -m pytest tests/test_render.py -ra` and
  `.venv/bin/python -m ruff check ground_truth tests`. Both clean. If any test is skipped, say so
  loudly — "tests pass" with a skip is a failure.

## Steps
1. Restate ≤8 bullets. Confirm you are implementing the settled decisions as written, and ask only
   what the tag census leaves open in D-10.1. **Wait for my D-10.1 answers before writing the pixel
   path.** You may write the non-pixel scaffolding (input list parsing, id hashing + collision check,
   instance ordering, manifest/back-map writers) and their tests in the meantime.
2. Write `ground_truth/render.py` (create the package dir; do not import `gt_schema.py` — 10a is a
   separate session and may not exist yet).
3. Write `tests/test_render.py`; run pytest + ruff until green.
4. Record each decision as a comment next to the code it governs (ID + chosen policy + one line of why).
5. Add a docstring header telling **me** how to run it, in the style of
   `scripts/preview_ct_axial.py` (human-run; view images in a standalone viewer, never in the
   Claude-connected IDE).
6. Confirm `git status` shows no PNG, `.dcm`, `.tar`, render manifest, or back-map staged; extend
   `.gitignore` only if the chosen output dir isn't already covered.
7. Show me the diff, then make **one** commit naming the phase (e.g. `Phase 10b: deterministic
   DICOM→PNG middle-frame renderer (D-10.1, D-10.2, D-10.8)`). Tick nothing in the README phase
   checklist — Phase 10 isn't done until 10e. Do not push.

## Done when
- `pytest tests/test_render.py -ra` green with zero skips; `ruff check` clean.
- The rerun test proves byte-identical PNGs + identical sha256s.
- Every settled decision above (D-10.2 + hash recovery path, forced RGB, D-10.8 input list, frame
  selection + fallback provenance, the two-file output split) is implemented and logged as a comment
  next to the code it governs, and D-10.1 is answered by me from the census and logged the same way.
- The back-map path is gitignored **and** hook-denied, added in this commit, with nothing loosened.
- One commit naming Phase 10b; `git show --stat` proves no PNG, `.dcm`, `.tar`, manifest, or back-map
  was committed.
- You have read no `.dcm`, no render, no `gt.csv` — and you did not run the renderer.

## Non-goals
No `gt_schema.py` (10a), no Tesseract seeding (10c), no review server/UI (10d), no
`build_gt.py`/`gt.csv`/`gt.csv.sha256` (10e). No harness change, no README phase tick, no new
dependency, no `git push`.
