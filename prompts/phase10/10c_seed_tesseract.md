> **BEFORE STARTING (2026-08-05):** 10b is committed (`bd37527`) but **D-10.1 (windowing) is still
> open**, so `ground_truth/render.py` raises `WindowPolicyUnsettled` and **no render exists yet**.
> 10c can be built and tested against synthetic PNGs — which is how it should be built anyway — but
> it cannot be *run* on real renders until D-10.1 lands. See the STATE block at the top of
> `10b_render.md` for my next actions, and the "Carry-overs" block in `10d_review_ui.md` for the
> safety lessons that apply here too (silence pydicom/Pillow warnings before any read; a library's
> warning path is a PHI egress path; assert on the artifact, not the label).

Phase 10c from plan.md — build `ground_truth/seed_tesseract.py`: run LOCAL Tesseract over the
Phase-10b renders to pre-fill boxes + strings as a SEED that the human corrects in 10d.

## Rule zero — this is a PHI-touching script (CLAUDE.md §0)
You WRITE it. I RUN it. You never see its output.
- Never `Read` a render (`.png`), a `.dcm`, `gt.csv`, or any seed file this script writes — not to
  "check the format", not once. No Bash command that pipes render pixels, OCR text, or seed contents
  back into your context.
- Everything the script emits to stdout is PHI-free aggregates only: per-image box counts, confidence
  distribution (min/median/max, histogram), image dimensions, totals, the Tesseract version string,
  the pinned config string. Never a token string, never a box's contents, never a non-hashed filename.
- Tesseract only, because it is local, neutral (never a candidate engine, so seeding cannot bias the
  bake-off), and its errors are obvious garbage rather than plausible near-misses. No cloud/network
  engine, no GPT-4o, no VLM, no HTTP call anywhere in this file — that would be PHI egress.
- Assume the script crashes on real data: no traceback, log line, or error message may interpolate OCR
  text, file contents, or a DICOM tag. Report exception *type* only, like
  `scripts/scan_text_poscontrol.py`.

## Context — read these, nothing else
- plan.md: the Phase 10 header gate, the 10c sub-step (the work), and 10d/10e — read those two only
  to know what CONSUMES the seed; do not build any of it. Plus §0 operating principles and the Phase
  10 acceptance criteria + review checklist.
- `ground_truth/render.py` (10b — built and committed): the interface you code against. Read the
  committed code for the renders dir, the `image_id` scheme (the hashed PNG stem) and the exact
  PHI-free render manifest path + field names. Read-only — do not modify it.
- `ground_truth/gt_schema.py` (10a — built and committed): the frozen `gt.csv` column spec and bbox
  validation the seed eventually flows through in 10e. Read-only — do not modify it.
- CLAUDE.md §0, §3, §7, §8, §9.
- `scripts/scan_text_ct_axial.py` and `scripts/scan_text_poscontrol.py` — the existing pytesseract
  call shape. Reuse it: `pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)`,
  iterate `data["text"]`/`data["conf"]`, parse conf as float with a `-1.0` fallback. The DICT also
  gives `left/top/width/height` — convert to `(x0, y0, x1, y1)` pixels, top-left origin, matching
  `OCRWord` (`harness/contract.py:26-47`) and `GTToken` (`harness/contract.py:96-118`). Same bbox
  convention, no exceptions.
- `harness/contract.py:121-139` — `normalize()` is FROZEN and applied only at match time. The seed
  stores Tesseract's raw string with zero cleanup.
- `tests/synthetic.py` — `FAKE_TOKENS` (includes `CMFN`-style fakes), `make_blank_image`,
  `make_synthetic_image`. All tests use these; never real data.
- `.gitignore`, `.claude/hooks/block_phi_read.py`, `.claude/settings.json` (`permissions.deny`),
  `pyproject.toml`.

## Environment facts already verified — do not re-derive, do act on them
1. `ground_truth/` exists; 10a (`gt_schema.py`) and 10b (`render.py`) are BUILT and COMMITTED. So the
   10b→10c interface is already pinned in code — read it, do not re-negotiate it: the renders dir,
   `image_id` (the hashed PNG stem, settled by 10b as D-10.2) and the PHI-free render manifest of
   `(image_id, frame_idx, w, h, sha256)`. Read 10b's committed `render.py` / render
   manifest writer for the exact paths and field names rather than assuming them; `image_id` here is
   whatever that code produces and you do not pick it. Still take the renders dir and manifest paths
   from a CLI arg with no silent default, and fail loud with an actionable message if either is
   missing or empty. Never invent renders.
2. The `tesseract` binary is **NOT installed here** (`which tesseract` is empty;
   `pytesseract.get_tesseract_version()` raises `TesseractNotFoundError`). `pytesseract` IS in
   `.venv` but is NOT declared in `pyproject.toml` — installed ad hoc, used only by `scripts/`. The
   sandbox network allowlist is PyPI/HuggingFace/GitHub only, so installing the binary is my step,
   not yours. Consequences: the script must detect a missing binary and exit with a clear setup
   instruction rather than a stack trace; and you must decide WITH ME whether `pytesseract` becomes a
   declared dependency (e.g. a pinned `[project.optional-dependencies]` extra alongside
   `doctr`/`paddle`/`easyocr`) or stays ad hoc. Do not add it to core `dependencies`.
3. Nothing currently gitignores or hook-blocks JSON under `ground_truth/`: `.gitignore` covers only
   `ground_truth/gt.csv` and `ground_truth/*.csv`, and `PHI_NAME`
   (`.claude/hooks/block_phi_read.py:27`) matches only `gt.csv` / `ground_truth*.csv` /
   `raw_response`. Whatever render/back-map patterns 10b added cover 10b's own outputs, not the seed
   — re-read `.gitignore` and the hook yourself to confirm the gap is still open before closing it.
   The seed files contain read-back PHI and need the same treatment as `gt.csv`.

## SETTLED DECISIONS — implement exactly these, log each beside the code it governs
Prompt-level decisions for 10c, numbered `D-10c.*` so they cannot be confused with plan.md's
`D-10.*`. Settled by me 2026-08-05. Do not re-litigate or "improve" them; if one is impossible as
written, stop and tell me rather than substituting your own.

- **D-10c.1 — RESOLVED: one JSON per image, `ground_truth/seed/<image_id>.json`.** Not one table.
  Same reasoning as 10d's review records: a killed run leaves only complete, valid files; 10d opens
  exactly one small file per image; two writes can never contend. Per-token schema: raw `text`,
  `bbox` as `(x0, y0, x1, y1)`, `confidence`, `label` (see D-10c.2).

- **D-10c.2 — RESOLVED: every seed token gets `label = "PHI"` by default.** Tesseract cannot know a
  token's label, so the default takes the conservative direction (redact rather than expose). That
  default is only acceptable paired with the two 10d requirements below — a defaulted label must
  never be mistakable for a verified one:
  - **Reviewed-state is visible, never inferred.** 10d already separates `accepted` / `edited` /
    `deferred` from "not yet reviewed"; that state must be visible at a glance per image and in a
    progress view across images, so an image still carrying default-PHI labels can never look like
    one whose labels I confirmed. Record nothing in the seed that implies review happened.
  - **Flipping PHI→KEEP is one keystroke and must not require me to consult a list by hand.** Review
    volume is high; per-token allowlist lookup by eye is the bottleneck to remove. The mechanism is
    10d's to build — its input is settled by **D-10c.2a** below.
  Both bullets are 10d work. **Do not build them here.** They are recorded because D-10c.2's default
  is only safe under them; carry them into the 10d prompt.

- **D-10c.3 — RESOLVED: header fields exactly as listed in D-10c.4, and `w`/`h` are KEPT.** They are
  not convenience: they are needed to assert every bbox is in-bounds, to verify the upscaled buffer
  is exactly `upscale_factor ×` the render, and by 10d to size its canvas.

- **D-10c.4 — RESOLVED: keep EVERY token; no confidence threshold, no minimum length.** The
  `CONF=50, MINLEN=3` in `scripts/` existed because those scripts *counted* detections. Here the seed
  is what the human sees, so a dropped faint token means no box appears and that token silently never
  reaches `gt.csv`. Clutter is handled at DISPLAY time in 10d (dim / toggle low-confidence boxes) —
  never by dropping rows in 10c. Dropping empty-text rows is parsing, not filtering (see
  `row_filter`).

  The pinned configuration and provenance block, recorded in EVERY seed file:

  ```
  tesseract_cmd_config = "--oem 1 --psm 11 -c load_system_dawg=0 -c load_freq_dawg=0 -c thresholding_method=0 -c invert_threshold=0.7 --dpi 300"
  lang                 = "eng"
  upscale_factor       = 2
  upscale_filter       = "PIL.Image.Resampling.LANCZOS"
  coord_space          = "original render pixels (x / upscale_factor, floats, unrounded)"
  row_filter           = "level == 5 AND text.strip() != ''"   # no conf threshold, no MINLEN
  conf_parse           = "float(conf), fallback -1.0, kept as-is (never dropped)"
  env                  = {"LC_ALL": "C", "LC_NUMERIC": "C", "OMP_THREAD_LIMIT": "1"}
  timeout_s            = 60
  tesseract_version    = <full string from `tesseract --version`>
  leptonica_version    = <from same>
  pytesseract_version  = "0.3.13"
  pillow_version       = <PIL.__version__>
  tessdata_dir         = <explicit --tessdata-dir path>
  tessdata_variant     = "best" | "fast" | "standard"
  traineddata_sha256   = <sha256 of the eng.traineddata actually loaded>
  ```

  Notes that constrain the implementation:
  - **The upscale is the ONE pixel operation 10c may perform**, and only because a pure resize is
    exactly invertible: boxes come back as `coord / upscale_factor`, unrounded. `OCRWord.bbox` and
    `GTToken.bbox` are both `tuple[float, float, float, float]`, so unrounded floats are
    contract-conformant — do NOT round, and add NO other preprocessing (no threshold, no denoise, no
    contrast). 10b owns rendering (CLAUDE.md §7).
  - `--dpi 300` is a pinned *claim*, not a measurement, and stays pinned even though the 2× upscale
    changes true effective DPI. Never compute it per image.
  - `-c load_system_dawg=0 -c load_freq_dawg=0` disable dictionary correction so ID-shaped strings
    are not pulled toward English words. `thresholding_method=0` pins Otsu (Tesseract 5 also ships
    Sauvola / LeptonicaOtsu). `invert_threshold=0.7` pins the default explicitly — it governs
    light-on-dark inversion, the common case for burned-in overlays, so it is the first knob to
    revisit if the seed under-detects, and changing it re-invalidates every seed already written.
  - `env` must be applied to the Tesseract subprocess. `OMP_THREAD_LIMIT=1` is a determinism
    requirement, not a perf tweak; `LC_*=C` keeps confidence parsing locale-independent.
  - `pytesseract.get_tesseract_version()` returns the version number only. `tesseract_version` and
    `leptonica_version` come from the full `tesseract --version` stdout — that output carries no PHI,
    so capturing it is fine. Capture once per run, not per image.
  - `row_filter`: `image_to_data`'s DICT `level` key is the hierarchy level and `5` is word level;
    levels 1–4 (page/block/para/line) are structure, not tokens.

- **D-10c.5 — RESOLVED: `pytesseract` becomes a declared dependency.** Other people at Gradient will
  run this harness, so a fresh clone must be able to install it. A pinned optional extra, named to
  match the existing engine-named extras:

  ```
  tesseract = ["pytesseract==0.3.13"]
  ```

  Never in core `dependencies`. The extra's comment block must state — the way `[doctr]` documents
  its CPU-torch preinstall — that pip installs the *wrapper only*: the `tesseract` binary, Leptonica,
  and the `eng.traineddata` are system-level and installed separately, and THOSE are what determine
  the seed (see the D-10c.4 provenance block). Do not touch `scripts/`' existing ad-hoc usage.

- **D-10c.6 — RESOLVED: manifest-driven, plus a bidirectional disk sweep, plus a re-hash preflight.**
  - **Iterate the manifest**, in sorted `image_id` order — the manifest is the authority on which
    images get seeded.
  - **Assert `set(disk PNG stems) == set(manifest image_ids)` and error on EITHER direction** — a
    manifest row with no PNG, and a PNG on disk absent from the manifest, are both loud failures. Not
    a warning, not a skip. (~5 lines over plain manifest iteration, and it closes both directions
    instead of one.) Glob `*.png` only; report the two difference SETS as counts and `image_id`s —
    `image_id` is a hash, so it is safe to print, unlike a filename.
  - **Re-hash every PNG and compare to the manifest's `sha256`.** Run this in a PREFLIGHT over the
    whole set, before any seed file is written, so a stale or truncated render aborts the run instead
    of leaving a half-seeded directory whose files silently came from different pixels.
  - **Manifest format: CSV — already fixed by the committed 10b, not re-chosen here.**
    `ground_truth/render.py` writes `<out-dir>/render_manifest.csv` with
    `MANIFEST_COLUMNS = ("image_id", "frame_idx", "w", "h", "sha256", "fallback_used")`, LF-pinned,
    and `fallback_used` serialized as `int(bool)` → `0`/`1`.
    **Do NOT write your own parser: import and reuse `ground_truth.render.read_manifest`**, which
    already validates the header against `MANIFEST_COLUMNS` and returns
    `image_id -> (frame_idx, w, h, sha256, fallback_used)` with `image_id`/`sha256` as `str`. One
    caveat to handle in 10c: `read_manifest` returns `{}` for a MISSING file rather than raising, so
    check existence and non-emptiness yourself and fail loud — do not let a missing manifest read as
    "zero images."
    Never substitute pandas here. It type-infers, and an all-digit 8-hex-char `image_id` (e.g.
    `12345678`) would be coerced to `int64` and silently break the join to the PNG stem.
  - `--renders-dir`, `--manifest`, `--out-dir` are CLI args with no silent default (as already
    required) — note 10b's own defaults are `renders/gt` and `<out-dir>/render_manifest.csv`, which
    is what I will pass, not what you may assume. `image_id` is NOT yours to pick — 10b's D-10.2
    fixed it as `sha256(f"{SOPInstanceUID}|{frame_idx}".encode("utf-8")).hexdigest()[:8]`, produced
    by `compute_image_id`.

- **D-10c.7 — RESOLVED: support BOTH skip and overwrite; ask when it is ambiguous.** `--on-existing
  {skip,overwrite}` with NO default. If `--out-dir` already contains seed files and the flag was not
  passed, prompt on stdin for skip / overwrite / abort. If stdin is not a TTY, do NOT guess — fail
  loud telling me to pass the flag, so batch runs stay explicit.
  **Guard that makes `skip` safe:** every seed file already carries the full D-10c.4 provenance block,
  so before skipping an image, compare the existing file's provenance to the current run's. If they
  differ **abort loudly** — never skip past it. Otherwise a config or version change would leave a
  seed directory silently mixing two provenances, which is exactly the invalidation CLAUDE.md rule #9
  exists to prevent.

- **D-10c.8 — RESOLVED: stdout aggregates AND a PHI-free `seed_summary.json` I may read.**
  - Path: `ground_truth/seed_summary.json` — deliberately OUTSIDE `ground_truth/seed/`, because the
    seed dir is about to be gitignored + hook-denied + added to `permissions.deny`. **Scope those
    deny patterns to the seed DIRECTORY, not to `ground_truth/*.json`**, or they will block the one
    file the summary exists to let me read. Verify that after writing them.
  - Contents, PHI-free by construction: the full D-10c.4 provenance block, total images, total
    tokens, per-image box counts keyed by `image_id` (a hash), the confidence distribution
    (min/median/max + histogram), image dimensions, and preflight results (set-diff counts, hash
    mismatches, timeouts). **No token text, no filenames, no strings read from any image, ever** — the
    same bar as stdout.
  - Gitignore it: it is regenerable, and a per-image count table is not something to publish. Flag it
    to me if you think it should be committed as provenance instead — that is my call, not yours.

- **D-10c.9 — RESOLVED: a timed-out or failed image is SKIPPED, and the failure is recorded in
  `seed_summary.json`.** `timeout_s = 60` will fire on some image eventually. That image gets no seed
  file, the run continues, and the summary gets a `failures` list of `{image_id, exception_type}` —
  **exception type only, never a message, traceback, or filename** (CLAUDE.md §0). Consequences that
  are part of this decision, not optional polish:
  - **Do NOT write a placeholder seed file with a `status: "failed"` field.** Its existence would make
    `--on-existing skip` (D-10c.7) stop retrying that image, turning a transient timeout into a
    permanent hole. No file = the next `skip` run retries it automatically. That self-healing is the
    reason skip is the right choice here.
  - **10d must treat "no seed file" as distinct from "seed file with zero tokens."** They are not the
    same thing: zero tokens is a legitimate, expected result on the blank ct_axial negative-control
    frames, so a failed image that silently renders as "blank, looks fine, OK" would write a
    no-tokens row into frozen `gt.csv` on exactly the axis the hallucination floor depends on. 10d
    reads the `failures` list and surfaces those images distinctly. **Do not build that here** —
    carry it into the 10d prompt alongside the D-10c.2 bullets.
  - A timeout is load-dependent, so it is the one thing that can break "same renders → byte-identical
    seed dir." Recording failures is what makes that visible instead of silent; say so in the
    determinism comment.

- **D-10c.2a — RESOLVED: a SEPARATE step emits the per-image known-good list. Not 10c, not 10d.**
  "Don't make me check the allowlist by hand" needs per-image known-good values, i.e. a join of
  `image_id` → 10b's `render_backmap.csv` → `manifest.csv`. That join belongs to neither 10b nor 10c
  today, and it is not going into either: a later small script (working name
  `ground_truth/build_allowlist.py`) performs it ONCE and writes
  `ground_truth/allowlist/<image_id>.json`; 10d just reads that file.
  - Why: it confines the back-map + `manifest.csv` PHI join to one auditable place instead of
    widening 10c's PHI surface (which is renders-only) or 10d's (which would then re-read the
    manifest on every launch and re-join per image at review time).
  - **Do not build it in 10c.** 10c still never touches `manifest.csv` and never reads the back-map.
    Carry this into the 10d prompt, and give `ground_truth/allowlist/` the same gitignore +
    hook-deny + `permissions.deny` treatment as `ground_truth/seed/` when it is built.
  - Rejected: a purely lexical/shape-based hint. It is PHI-free and free to build, but it cannot
    distinguish a valid accession from a misread one — which is exactly the false-redaction call
    being made — so it would be a guess dressed as ground truth.

- **D-10c.2b — RESOLVED: `tessdata_variant = "best"`** (tessdata_best, the LSTM float models).
  Accuracy over speed: this is a one-time seeding run over a few hundred images, so `fast`'s
  int8 speedup buys nothing and costs reads. `best` is **not** what `apt-get install tesseract-ocr`
  installs (that is `standard`) — it is a separate download from the `tessdata_best` repo, so this
  changes the install step; the `traineddata_sha256` in the provenance block is what actually pins
  it, and a peer reproducing the seed must match that hash, not just the variant name.
  - **The script still must not guess.** `--tessdata-dir` and `--tessdata-variant` both stay
    REQUIRED with no default. The variant is a human claim that the script records but cannot
    verify, so it has to be asserted explicitly on every run; `best` is the expected value, not a
    fallback.
  - Switching variants later re-invalidates every seed already written (CLAUDE.md rule #9) — the
    same class of change as a `gt.csv` edit.

(No decisions remain open for 10c.)

## Build requirements
- Deterministic: sorted input order, the pinned `env` applied to the Tesseract subprocess, and the
  full D-10c.4 provenance block recorded IN every seed file. A bump to ANY of Tesseract, Leptonica,
  the `eng.traineddata`, `pytesseract`, or Pillow changes the seed — the same class of invalidation as
  an engine bump (CLAUDE.md rule #9), which is why each is recorded with a version and the
  traineddata with a hash. Say so in a comment and in the emitted summary. Rerunning on unchanged
  renders must produce byte-identical seed files.
- Idempotent / resumable: re-run behavior per D-10c.7 (`--on-existing`, prompt when ambiguous,
  provenance-mismatch abort), and a killed run leaves valid files, never a half-written one. Write
  atomically (temp file + `os.replace`).
- Log every settled decision as a comment at the point in code it governs, plus a module docstring
  opening with the "RUN THIS YOURSELF — never let Claude read its output" banner used in
  `scripts/scan_text_ct_axial.py`.
- Add the seed dir to `.gitignore`, to the hook's deny patterns, and to `.claude/settings.json`
  `permissions.deny`. **Scope every one of those patterns to `ground_truth/seed/` specifically** — a
  broader `ground_truth/*.json` would also block `seed_summary.json`, the one file D-10c.8 exists to
  let me read. Gitignore `seed_summary.json` too, but do NOT hook-deny it. After writing the
  patterns, verify both directions: a seed file is denied, and the summary is readable.
  ADD patterns only — never loosen, reorder, or remove an existing rule (CLAUDE.md §6.4). Do not add
  `ground_truth/review/` coverage; that belongs to 10d.
- Declare the `tesseract` extra in `pyproject.toml` per D-10c.5, with the comment block explaining
  that pip carries the wrapper only.

## Tests — synthetic only, none skipped
Create `tests/test_seed_tesseract.py` using `tests/synthetic.py`. Cover:
- Bbox conversion from `image_to_data` DICT geometry to `(x0,y0,x1,y1)` with `x0<x1`, `y0<y1`, inside
  image bounds — pure-function tested with a fabricated DICT, no Tesseract needed.
- Upscale coordinate round-trip (D-10c.4): a fabricated DICT in upscaled space maps to
  `coord / upscale_factor` exactly, unrounded, and the upscaled buffer's dimensions are exactly
  `upscale_factor ×` the source `w`/`h`.
- No filtering (D-10c.4): a fabricated DICT containing a `conf=8`, 1-character, level-5 token keeps
  that token; level 1–4 and empty-text rows are dropped.
- Every seeded token carries `label == "PHI"` (D-10c.2), and nothing in the file implies review state.
- Raw text preserved: no `normalize()`, no strip-and-store, no case folding.
- Seed-file schema round-trips; the full D-10c.4 provenance block is present and complete — a missing
  field fails loud rather than writing a seed file with unknown provenance.
- Determinism: two runs over the same synthetic input produce identical bytes.
- Preflight (D-10c.6), with a synthetic renders dir + hand-written CSV manifest: a manifest row with
  no PNG errors; an extra PNG not in the manifest errors; a PNG whose bytes don't match the manifest
  `sha256` errors — and in all three cases NO seed file is written, proving the preflight ran first.
- Manifest parsing (D-10c.6): an all-digit 8-character `image_id` (e.g. `12345678`) survives as a
  `str` and still joins to its PNG stem — the regression pandas would introduce. Plus: a MISSING
  manifest fails loud rather than reading as zero images (`read_manifest` returns `{}` there).
- Re-run behavior (D-10c.7): `--on-existing skip` leaves an existing file untouched; `overwrite`
  replaces it; a seed file whose provenance block differs from the current run aborts under `skip`
  instead of being skipped; a missing flag with a non-TTY stdin fails loud rather than defaulting.
- Failure recording (D-10c.9): a simulated per-image failure writes NO seed file for that image, does
  not abort the run, lands in the summary's `failures` list as `{image_id, exception_type}` with no
  message or traceback, and is retried on a subsequent `--on-existing skip` run.
- `seed_summary.json` (D-10c.8) is valid JSON, carries the provenance block and the counts, and
  contains no token text — assert against a synthetic run whose fake tokens are known, so the test
  can prove those exact strings are absent from the file.
- **Synthetic positive control (CLAUDE.md §8).** A "0 text regions" result is meaningless until the
  detector is validated. Generate a PHI-free image with fake burned-in text via
  `make_synthetic_image` (tokens like `CMFN-00421`) and assert the seeder finds boxes on it; pair it
  with `make_blank_image` asserting zero/near-zero. This is PHI-free, so you may generate and run it
  yourself. If the `tesseract` binary is absent, this test must FAIL LOUD or be explicitly reported
  as not-run — never silently skipped into a green run.
Verify with `.venv/bin/python -m pytest tests/test_seed_tesseract.py -ra` and
`.venv/bin/python -m ruff check ground_truth tests`. Report the real result including anything that
did not execute because the binary is missing.

## Out of scope — do not touch
`ground_truth/render.py` (10b) and `gt_schema.py` (10a) — both built and committed: read or import
them, never modify them. Also `review_gt.py` / `review_ui.html` / `ground_truth/review/` (10d),
`build_gt.py` / `gt.csv` / `gt.csv.sha256` (10e), anything under `harness/`, `normalize()`,
`iou_thr`, and every existing script in `scripts/`. If 10c looks blocked on one of those, stop and
tell me — do not build it.

## Done when
- `ground_truth/seed_tesseract.py` exists, runs nothing on import, and its stdout is PHI-free
  aggregates only.
- Tests green with nothing silently skipped; ruff clean.
- Every settled decision logged in a comment beside the code it governs.
- Seed dir gitignored + hook-denied + in `permissions.deny`; no existing safety rule weakened.
- One commit naming the phase (e.g. `Phase 10c: Tesseract seed for GT review`). Do not push.
- You have read no `.dcm`, no render, no seed file, no `gt.csv`.

## Report
What was built, the decisions as settled, the test result verbatim (including anything that did not
run), and the exact commands I need to run — starting with installing the `tesseract` binary.
