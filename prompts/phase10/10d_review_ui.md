Phase 10d from plan.md — build the human review-and-correction UI. This is the single most
PHI-sensitive tool in the repo: 🔴 PHI-DISPLAYING. You write it; I (Arnav, sole annotator) run it.
Scope is 10d only — 10b (`render.py`) and 10c (`seed_tesseract.py`) are built and committed: read or
import them, never modify them. Do not build or stub 10e (`build_gt.py`).

## Rule zero (overrides everything below)
You never see PHI — **and neither does any subagent you spawn** (see the propagation block in
`## Working rhythm`). For this whole session:
- Do NOT Read or open: any `.dcm`, anything under `renders/`, any `.png/.jpg/.tif/.npy`,
  `ground_truth/gt.csv`, `ground_truth/review/*.json`, anything matching `raw_response`, any
  `render_backmap*.csv`, `manifest.csv`, or `gt_sample_*.csv` (the last three are a Limited Data Set —
  UIDs, full study dates, institution pseudonyms).
- Do NOT launch a browser, screenshot the UI, or invoke the `run` skill on it. I look at the images;
  that is the entire premise (CLAUDE.md §0).
- Do NOT run the server against real renders. Run it only against synthetic PHI-free fixtures you
  generate (fake images, fake tokens like `CMFN-00421`).
- Never print token text, never weaken a safety rail.

## Read first (source + docs only)
- plan.md: the full 10d sub-step (it is detailed — read all of it), the Phase 10 header gate, the
  decisions block, the acceptance criteria and review checklist, and §0 operating principles.
- CLAUDE.md §0 (golden rule), §6 (hard stops), §7 (safe-working conventions), §9 (enforcement layers).
- `.gitignore`, `.claude/hooks/block_phi_read.py`, `.claude/settings.json` (`sandbox` +
  `permissions.deny`).
- `scripts/build_seed_groups.py` — the existing `image_id,group` builder you consume, do not rewrite.
- `harness/contract.py:96-118` — `GTToken`: bbox is `(x0, y0, x1, y1)`, axis-aligned, TOP-LEFT origin,
  in PIXELS. The review JSON must use exactly this convention so 10e maps 1:1 with zero
  re-interpretation.
- `tests/synthetic.py` + `tests/conftest.py` — reuse the existing synthetic fixture factory; do not
  invent a second one.

Current repo state: 10a (`ground_truth/gt_schema.py`), 10b (`ground_truth/render.py`) and 10c
(`ground_truth/seed_tesseract.py`) are built and committed, so every input 10d consumes is already
pinned in code. Read the committed 10b/10c code for the exact paths and field names — 10b's renders
directory and PHI-free render manifest (`MANIFEST_COLUMNS`, `render.py:86`), and 10c's per-image seed
JSON — and code against them; do not re-derive or re-negotiate them. `image_id` is whatever 10b
produces; you do not pick it. Document the consumed input schema in a module docstring at the top of
`review_gt.py`. Do not implement or modify the renderer or the seeder.

### Inputs are MULTIPLE (render_dir, seed_dir) pairs — not one pair. This is a hard requirement.

`gt_v2` is 202 images that were rendered and seeded in several batches, into several directories
(`renders/gt` + `ground_truth/seed`, `renders/pilot` + `ground_truth/seed_pilot`, `renders/v2` +
`ground_truth/seed_v2`, and others). **Do not hardcode a single pair.** Take a repeatable
`--set <renders_dir>:<seed_dir>` flag (or equivalent) and build the review set as the **union**.

- Every pair is validated at startup: the renders dir has a `render_manifest.csv`, and every
  `image_id` in it has a seed JSON. A missing seed is a **startup failure**, not a silently skipped
  image — a silently dropped image never reaches `gt.csv`, the exact failure mode D-10c.4 exists to
  prevent.
- A duplicate `image_id` across two pairs is a **hard error naming both paths**. `image_id` is 8 hex
  chars of sha256 (`render.py:105`), so a collision is unlikely but possible, and merging two images'
  ground truth silently is unrecoverable. Same argument as `render.py:_check_ids`.
- The review output directory is one flat `ground_truth/review/<image_id>.json` regardless of which
  pair the image came from — `image_id` is globally unique by the check above.

### Stratum comes from a PHI-free groups CSV — reuse the existing builder, do not write a new one

The per-stratum gate below needs `image_id -> stratum`. That mapping already exists as a solved,
PHI-free artifact: `scripts/build_seed_groups.py` joins the (PHI-adjacent) back-map to the sample
manifest and writes **only** `image_id,group`. Take a required `--groups <csv>` flag with those two
columns. Do not read the back-map, `manifest.csv`, or `gt_sample_v2.csv` from `review_gt.py` — the
groups CSV is the entire interface, and keeping it that way is what keeps 10d PHI-free at rest.
An `image_id` absent from the groups CSV gets the default gate and is **counted and reported** at
startup, never silently defaulted.

## Build
`ground_truth/review_gt.py` + `ground_truth/review_ui.html`.

> Read `## RESOLVED DECISION RECORD` near the bottom of this file before writing a line — it fixes the
> port, the keyboard shortcuts and auto-advance semantics, the `deferred` reason enum, the `round`/
> `timestamp` fields, and **the view-time filter (D-10d.A = A3), which IS in scope**. It overrides
> anything below it that reads as open. Where the narrative in `## DECISIONS` disagrees with the
> record, the record wins; the narrative is history, not instruction.

Per image the UI shows the render TWICE, side by side:
- left: the original, untouched;
- right: the same image with the seed boxes drawn and each read string beside its box.

Both panes are required — an overlay hides the pixels underneath it, which is exactly where a missed
or mis-bounded token lives. Do not collapse this into one toggleable pane.

One decision per image: OK (accept every seed as-is) or Not OK (drop into edit mode).

Edit mode, minimum operation set: drag on the canvas to draw a new box; click a box to select it; edit
the selected box's text; nudge and resize it; delete it; set its label `PHI` or `KEEP` (binary — there
is no `OTHER`, do not add one).

Every image ends in exactly one of three states: `accepted`, `edited`, `deferred`. "Not yet reviewed"
and "reviewed and fine" must never be representable as the same thing — absence of a file means
unreviewed, and nothing else.

Resumable: the record is written after EVERY image, not at end of session, so closing the browser
costs at most one image. Re-opening a reviewed image shows the saved state, not the raw seed.

## The view-time filter (D-10d.A = A3 — RESOLVED, build it)

The seed is maximal-recall on purpose and stays frozen; the **filtering happens here, at display
time.** Read `### D-10d.A` below for the measurement that forced this. The semantics that matter:

- **Hidden means EXCLUDED from `gt.csv`, not "accepted".** A token below the gate is not written into
  the review record's token list. It requires no click. The reviewer sees exactly the reduced set —
  identical workload to seed-time gating. The only difference is that the token still exists on disk,
  so an over-filtered stratum is fixed by moving a slider instead of re-seeding everything.
- **Per-stratum defaults, loaded from a config dict at the top of the module — not inline literals.**
  The `UNCOUNTED` rows below will be recalibrated once Arnav supplies human word counts, and that must
  be a one-line edit, not a hunt through the file.
- **`len >= 1` and `alnum` everywhere. There is NO length floor, at any stratum, ever.** One
  `ct_secondary_capture` frame's entire real content is `R` and `L`; a `len>=2` floor seeds it empty.
  `GTToken.label` is binary PHI/KEEP and a misread `L` that gets redacted **is** a false redaction —
  the headline metric. If you find yourself adding a length control, stop and re-read this paragraph.
- The confidence slider is **live and per-image adjustable**, defaulting to the stratum's value. A
  frame that looks suspiciously empty is diagnosed by sliding down, which is the whole point of A3.
- The **effective gate at decision time is recorded in the review JSON** (`gate: {conf, len, alnum}`)
  so a record always says which threshold produced its token list. Provenance, not an index — same
  rule as `timestamp` (record item 5): it must never reach a `gt.csv` column.

Defaults (calibrated 2026-08-07 against human word counts where they exist, deliberately loose where
they don't; `c<N>` = `confidence >= N`, always with `len>=1 + alnum`):

```
stratum                imgs   gate   basis
us_ge                    14    c10    ~55 real by eye, c10 keeps 52
us_philips               13    c30    34 real, c30 keeps 33
us_siemens               13    c30    17 real, c30 keeps 17
us_toshiba_canon          9    c10    UNCOUNTED - loose on purpose
us_samsung                5    c10    UNCOUNTED - loose on purpose
us_sonosite               2    c10    UNCOUNTED - loose on purpose
us_other                  1    c10    UNCOUNTED - loose on purpose
ct_secondary_capture      4    c0     no confidence floor discriminates
mg_2d                    44    c60    ~1.5 real vs 363.7 ungated
mg_tomo                  16    c60    blank stratum
ct_scout                 12    c60    blank stratum
ct_axial                 66    c60    blank control
(unknown stratum)          -    c0     default: never hide by accident

TOTAL: 199 images. The v2 render (2026-08-07) took 199 of 202; 3 series hard-failed on
AmbiguousFrameAxisError (one each in mg_tomo, us_philips, us_siemens). The `imgs` column is
context, not a check — do NOT assert these counts anywhere in code or tests.
```

The default for an unlisted or unmatched stratum is **`c0` (show everything)**. Erring toward more
boxes costs clicks; erring toward fewer silently deletes real text, which is the expensive error.

## Shape constraints (hard)
- Exactly two files, roughly 450–550 lines total. The A3 filter, the multi-pair input, and the groups
  join are what moved this up from 300–400. If you're past 550, cut features, not safety.
- `review_gt.py`: stdlib `http.server` only — serves `review_ui.html`, the PNGs, the seed data, and
  one save endpoint. `flask` is NOT a dependency and must not become one. No new heavy deps; do not
  touch `pyproject.toml` pins.
- `review_ui.html`: one page, one `<canvas>`, vanilla JS, image drawn under the boxes. No build step,
  no npm, no framework, no bundler.
- Bind `127.0.0.1` only, on an explicit port. It needs no auth precisely BECAUSE of that bind — so the
  bind must not be configurable to anything else. No `0.0.0.0`, no host flag.
- Fully offline: no CDN, no remote fonts, no telemetry, no outbound request of any kind. Every asset
  served from disk, inlined or local-path only. Make the page *incapable* of egress rather than
  arguing about whether a given request carried PHI.
- Nothing PHI to stdout: override `http.server`'s `log_message` to silence the per-request path log.
  Never put token text in a URL, query string, header, log line, error message, or traceback — key
  every endpoint by `image_id` (already a hash) only, and send token payloads in request/response
  BODIES. Ensure no handler exception can echo a request body to the terminal.

## Out
One JSON per image at `ground_truth/review/<image_id>.json`, holding the final token list (`text`,
`box`, `label`) plus `state`, `round`, `timestamp`, and — when `state == "deferred"` — `defer_reason`
and an optional `note` (see the decision record for the enum values and the never-print rule on
`note`). One file per image, not one CSV: resumability is free, a killed session
leaves valid files rather than a half-written table, and two writes can never contend. Write
atomically (temp file + `os.replace`) so a kill mid-write cannot corrupt a record.

## Close the safety-rail gap (do not skip, do not weaken)
These JSONs contain `token_text`, so they ARE PHI and get the same handling as `gt.csv`:
1. `.gitignore`: add `ground_truth/review/`. Today's `ground_truth/*.csv` does not cover `.json`.
2. `.claude/hooks/block_phi_read.py`: extend the deny patterns so a review JSON cannot be Read. Today
   `PHI_NAME` (line 27) is `(^|/)(gt\.csv|ground_truth.*\.csv|.*raw_response.*)$` — `.csv` only — and
   `PHI_EXT` has no `.json`. Widen it (e.g. cover `ground_truth/review/` and
   `ground_truth/**/*.json`) without narrowing anything already denied. Add a unit test that feeds the
   hook a synthetic `Read` of `ground_truth/review/abc123.json` and asserts a `deny` decision.
3. `.claude/settings.json` `permissions.deny`: add the matching `Read(./ground_truth/review/**)` entry.
4. The root-owned managed copy at `/etc/claude-code/hooks/phi_guard.py` must stay in sync but you
   cannot edit it — do not try. Flag it in your final report as an action for me.

## PHI-free summary (the only thing you ever read)
Add a summary mode (e.g. `--summary`) that reads `ground_truth/review/*.json` and emits ONLY:
- Progress: images total / accepted-as-is / edited / deferred / not yet reviewed.
- Seed quality, counts and percentages only: seeded boxes kept unchanged, text-corrected, deleted as
  spurious, and boxes added by hand.
- GT box geometry distribution: median and percentile (p10/p50/p90) box width and height in px, plus
  box height relative to image height.

The geometry block is a hard requirement, not a nicety: it is the measurement that settles whether
`iou_thr = 0.5` is right (EasyOCR's boxes measured ~1.9× taller than the text they bound). Do not
change `iou_thr` or anything in `harness/matching.py` in this session — just produce the number.

Never a token string, never a filename that isn't already a hash, never a box's contents.

## Tests (synthetic only, none skipped)
New `tests/test_review_gt.py`, built on `tests/synthetic.py` fixtures with fake tokens
(`CMFN-00421`-style). Required cases:
1. Zero-egress static check: parse `review_ui.html` and assert no `http://` / `https://` / `//`
   external asset reference, no `<link>`/`<script src>`/`@import`/`url()` pointing off-disk, and no
   `fetch`/`XMLHttpRequest`/`WebSocket`/`sendBeacon` target that isn't a same-origin relative path.
2. Server binds loopback only: assert the bind address is `127.0.0.1` and that no argument or env var
   can change it.
3. Request log silenced: start the server on an ephemeral loopback port, exercise the endpoints,
   capture stdout/stderr, assert empty — and specifically assert no request path appears.
4. No PHI in stdout: assert the fake token string never appears in server output, in any URL the
   client requests, or in `--summary` output.
5. Round-trip state: save `accepted` / `edited` / `deferred` records; reopening returns the saved
   state, not the seed; an unreviewed image has no file at all.
6. Atomic write survives a simulated kill mid-save (no partial/invalid JSON left behind).
7. Summary correctness on a synthetic review set: progress counts, seed-quality counts, and geometry
   percentiles all match hand-computed values.
8. The hook-deny test from the section above.
9. **Filter semantics (the one that protects the headline metric).** With a synthetic seed containing
   tokens at confidence 5 / 45 / 95 and a single-char `L`: at `c30` the conf-5 token is hidden and
   **absent from the saved record's token list** (assert on the written JSON — carry-over #3), the
   other two are present; sliding to `c0` and re-accepting restores it; the single-char `L` survives
   **every** gate value (assert across the full slider range — this is the no-length-floor guarantee).
10. **Per-stratum defaults resolve from the groups CSV**: an image in `mg_2d` opens at `c60`, one in
    `ct_secondary_capture` at `c0`, one absent from the groups CSV at `c0`, and the count of
    unmatched `image_id`s is reported at startup.
11. **Multi-pair input:** two `(renders, seed)` pairs union correctly; a pair whose seed JSON is
    missing for a manifest `image_id` fails at startup; a duplicate `image_id` across pairs raises an
    error naming both paths.

Put any `urllib`/`http.client` usage inside the test file and run it as `python -m pytest`. Do not run
inline `python -c "...urllib..."` — the PHI hook denies that command shape (correctly), and that is
not a reason to touch the hook.

Verify with:
```
python -m pytest tests/test_review_gt.py tests/ -q
ruff check .
git status --porcelain
```
`git status --porcelain` must show no `renders/` and no `ground_truth/review/` entries. I will
separately verify zero egress at runtime with `ss -ltnp` / `netstat` while the UI is open — write the
exact command for me to run in your final report.

## Working rhythm — run this as an agent team, you are the lead orchestrator

You are on **Fable** and you are the **lead orchestrator**. You own the plan, the two core files, the
merge, and the final report. Delegate everything that is genuinely independent; do the coupled work
yourself. Spawn agents with the `Agent` tool and an explicit `model` override.

### Rule zero propagates — this is the one that can actually hurt us

**A subagent inherits none of this prompt's context.** It will happily `Read` a render if its task
brushes near one, and that is an unauthorized PHI disclosure under a non-BAA API (CLAUDE.md §0) — the
hook is a backstop, not permission.

> **Every subagent prompt you write MUST open with this block, verbatim:**
>
> ```
> PHI RULE (overrides your task): Never Read, open, render, decode or print any .dcm, anything under
> renders/, any .png/.jpg/.tif/.npy, ground_truth/gt.csv, ground_truth/review/*.json, any
> render_backmap*.csv, manifest.csv, gt_sample_*.csv, or anything matching raw_response. Never launch
> a browser, screenshot, or run the review server against real renders. Never print token text. Never
> weaken a safety rail. Source code, tests, configs and docs only. If your task appears to require
> reading one of these, STOP and report that instead of doing it.
> ```

Also give every subagent: a **narrow scope, an explicit deliverable format, and a word budget**
(≤300 words for recon agents). Do not send an agent to "explore the codebase."

### Waves

**Wave 0 — you, alone, no agents.** Restate the plan in ≤8 bullets (files to create/edit, endpoints,
the state machine, filter semantics, summary fields, safety-rail edits, test list). **Every decision
is already made — see `## RESOLVED DECISION RECORD`. Do not re-surface the port, the shortcuts, the
defer reason, the round/timestamp fields, or D-10d.A/B for me to pick again; do not ask for a
go-ahead.** The one exception: if any part of the restatement contradicts the decision record or Rule
zero, stop there and say which, rather than resolving it yourself. Then keep going without stopping.

**Wave 1 — recon, 3 agents in PARALLEL, all `model: "sonnet"`, all read-only.** One message, three
tool calls. Each returns a `file:line — fact` list, no prose, no recommendations:
- **R1 — input contract.** From `ground_truth/render.py` and `ground_truth/seed_tesseract.py`: the
  render manifest columns, the seed JSON top-level keys, the per-token keys and their exact types, the
  `image_id` derivation, and how `--on-existing` names files. Deliverable: a table.
- **R2 — safety rails.** Exact current text and line numbers of the `PHI_NAME` / `PHI_EXT` patterns in
  `.claude/hooks/block_phi_read.py`, the `permissions.deny` list in `.claude/settings.json`, and the
  `ground_truth` lines in `.gitignore`. Deliverable: quoted current values + the precise insertion
  point for each. It does **not** edit anything.
- **R3 — test infrastructure.** The public API of `tests/synthetic.py` and the fixtures in
  `tests/conftest.py`: names, signatures, what each produces. Deliverable: signatures only. The point
  is that you reuse this factory rather than inventing a second one.

**Wave 2 — build, 2 agents in PARALLEL on DISJOINT files.** Never two writers on one file; that is
why no worktree isolation is needed here.
- **You (Fable), yourself:** write `ground_truth/review_gt.py` and `ground_truth/review_ui.html`. These
  two are tightly coupled — endpoint shapes, the filter, the state machine — and are the whole
  PHI-sensitive surface. **Do not delegate them.**
- **B1, `model: "sonnet"`:** the three safety-rail edits (`.gitignore`, hook patterns,
  `permissions.deny`) using R2's exact insertion points, plus the `git check-ignore -v` proof. Tell it:
  widen, never narrow; comments on their own line (carry-over #4); do not touch
  `/etc/claude-code/hooks/phi_guard.py`.

**Wave 3 — tests, 1 agent, `model: "opus"`, after Wave 2 lands.** All thirteen required cases in
`tests/test_review_gt.py`, built on R3's factory. Opus, not Sonnet: cases 9–11 (filter semantics,
per-stratum defaults, multi-pair) are where a plausible-looking test passes while asserting the wrong
thing, and carry-over #3 exists because exactly that already happened once in 10b. Give it the
"assert on the artifact, not the label" rule verbatim.

**Wave 4 — code review, 2 agents in PARALLEL, both `model: "opus"`, both adversarial.** Prompt each to
*find the defect*, not to bless the work, and to return `file:line — defect — why it matters`, with
"no findings" a permitted answer. Do **not** let either fix anything; you apply the fixes.
- **CR1 — PHI-egress lens.** Can any token text, note, or filename reach stdout, a URL, a query
  string, a header, a log line, a traceback, or an HTTP error body? Can the page make any outbound
  request? Can the bind be moved off `127.0.0.1` by any argument, env var or config? Is `note`
  excluded from `--summary`? Are the rails actually widened and not narrowed?
- **CR2 — correctness lens.** Filter semantics (hidden ⇒ excluded, not accepted; no length floor at
  any gate value; unlisted stratum ⇒ `c0`); atomic write; the save-completes-before-advance ordering;
  `←`/`→` reaching reviewed images; unreviewed ⇒ **no file**; multi-pair union and the duplicate
  `image_id` error; and whether each test would actually fail if the behaviour it names were broken.

**Wave 5 — you, alone.** Apply the fixes, re-run the full verification block, commit, write the final
report. State plainly what each reviewer found, what you fixed, and what you consciously did not.

### Two things not to do
- **Do not spawn an agent for work that is faster done inline.** Thirteen agents on a ~500-line
  two-file job is worse than five. The waves above are the budget; adding a sixth agent needs a reason
  you can state in one sentence.
- **Do not let a subagent's report stand as verification.** "R2 says the hook denies it" is not the
  test passing. Every claim that matters lands as a test you ran, or as a file you read yourself.

### Then, regardless of who wrote what
Record each resolved decision as a comment where the code governs it (why loopback-only means no auth;
why one JSON per image; why both panes; why the log is silenced; why hidden means excluded; why there
is no length floor).

## Carry-overs from the 10b / tag-census session (2026-08-05) — apply these, don't rediscover them

1. **A library's warning path is a PHI egress path.** Running `scripts/tag_census.py` printed a raw
   SOPInstanceUID to the terminal: this dataset contains UIDs longer than the 64 characters VR UI
   allows, and pydicom's validator warns with the offending value inline. Any script here that opens
   a `.dcm` — or any library that might echo input on a validation failure — must silence it *before*
   the first read, and must not rely on "I just won't paste that part". Reuse the exact helper now in
   `ground_truth/render.py` (`_silence_pydicom`) and `scripts/tag_census.py`:

   ```python
   warnings.simplefilter("ignore")
   logging.getLogger("pydicom").setLevel(logging.CRITICAL)
   pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
   ```

   For 10d the equivalents are: `http.server`'s request log (already required above), Pillow's
   `DecompressionBombWarning`/EXIF warnings, and any traceback rendered into an HTTP error body —
   the browser is a display surface, but the terminal and the response body are not.
2. **Non-conformant UIDs are normal in this dataset.** Never validate, truncate, or round-trip a UID
   as if it were conformant. `image_id` hashing is unaffected (sha256 of any length), but anything
   that echoes, parses, or bounds-checks a UID will produce a value-bearing error message.
3. **Assert on the artifact, not on the label.** In 10b, 43 tests passed with the renderer forced to
   always decode frame 0 — the manifest still *said* "middle frame". For 10d that means: reopening a
   reviewed image must be tested by comparing the rendered token boxes/text to what was saved, not by
   asserting a `state == "accepted"` field or that a file exists.
4. **Git has no inline comments.** A `.gitignore` line like `ground_truth/review/   # PHI` matches
   nothing — the comment becomes part of the pattern. Comments go on their own line, and every new
   rule gets a `git check-ignore -v` proof in your final report.
5. **The rails only gate the `Read` tool.** `cat`/`grep` on a UID-bearing file via Bash is caught by
   neither `permissions.deny` nor the hook's Bash patterns (pre-existing, same for `gt.csv`). When
   you add `ground_truth/review/` to the rails, say so explicitly rather than claiming the files are
   "denied"; closing the Bash gap is a separate decision for me, not a 10d change.
6. **Sync the managed twin.** `/etc/claude-code/hooks/phi_guard.py` is root-owned and needs the same
   new pattern; list it as an action item for me, do not attempt to edit it.
7. **Sidecars that cannot be re-derived get atomic writes.** A review JSON is the only record of
   human annotation work — write to `.part` + `os.replace`, never truncate in place.
8. ~~**Still open from 10b:** D-10.1 (windowing) is unsettled…~~ **SUPERSEDED 2026-08-06.** D-10.1 is
   resolved and implemented; `render.py` no longer raises `WindowPolicyUnsettled`.
9. **Positional references to images are unsafe — always exchange explicit `image_id`s.** A batch of
   human labels handed back as "the first 10 in the folder" silently permuted: the file manager
   natural-sorts, `ls | sort` does not, and the mislabeling was caught by luck. Anywhere the UI or
   `--summary` names an image to Arnav, name it by `image_id` — never an ordinal, never "image 4 of
   202". Position is a UI affordance; identity is the hash.
10. **The 2026-08-07 re-seed did not happen and is not coming.** Earlier drafts of this prompt said
    "synthetic fixtures only; real annotation waits for plan.md Phase E," because Phase E was going to
    delete `ground_truth/seed/` and re-seed once a winning gate config was found. **A3 replaced that
    plan** — the seed is now permanently unfiltered and the gate moved into this UI, so there is
    nothing to re-seed and nothing to wait for. Arnav annotates all 202 images with 10d as soon as it
    ships. Your obligations are unchanged: **you** still build and test against synthetic PHI-free
    fixtures only, and **he** runs it against the real renders (Rule zero).

---

## DECISIONS — RESOLVED 2026-08-06 (read this whole section; the evidence explains the calls)

> **⚠️ HISTORY, NOT INSTRUCTION.** Everything from here to `## RESOLVED DECISION RECORD` is the
> measurement narrative — retained because it is *why* the settled values are what they are, and
> because the 2026-08-07 pilot **overturned** part of it. Read it for reasoning; take orders only from
> `### D-10d.A` (resolved: A3) and the decision record. In particular the gate sweep immediately below
> was run on blank strata only and its conclusions were superseded — do not build to it.

### Original framing, raised 2026-08-06

Both come out of the first real seed run: 66 blank `ct_axial` frames produced **1,114 seed tokens**
(mean 16.9/image, median 8, max 265 — two frames hold ~37% of the noise; 6 images are already at 0).
Nothing is broken. `seed_tesseract.py`'s `row_filter` is `level == 5 AND text.strip() != ''`, i.e.
every non-empty detection at any confidence, which is the correct recall-first default for a seeder
(a human can delete a false box; they cannot recover one never proposed). Note Tesseract is the
**seeder**, not a benchmark candidate (CLAUDE.md §4 — neutral by design), so this noise costs
annotation labor only. It does **not** enter any engine's score and it is **not** the hallucination
floor; that gets measured when docTR / PP-OCRv6 / EasyOCR run on these same 66 blank frames.

### Measured gate sweep (`scripts/seed_gate_sweep.py`, run on the 66 blank `ct_axial` seeds)

Tokens surviving each gate, images with ≥1 in parentheses; alnum required in the grid:

```
single lever          conf>=35 alone: 406 (36.4%)   len>=3 alone: 143 (12.8%)
                      conf>=50 alone: 193 (17.3%)   len>=4 alone:  26 ( 2.3%)
                      alnum alone:    889 (79.8%)

combined              len>=1        len>=2        len>=3        len>=4
conf>=0             889 (59)      632 (54)      136 (38)       25 (13)
conf>=30            380 (51)      255 (45)       55 (25)         7  (5)
conf>=35            294 (50)      188 (44)       42 (22)         5  (4)
conf>=40            230 (48)      145 (41)       33 (18)         2  (2)
conf>=50            124 (40)       65 (32)         8  (7)        1  (1)
```

**Length dominates confidence.** `len>=3` alone removes more noise (143 left) than `conf>=50` alone
(193 left). The originally-proposed `conf>=35` alone leaves 406 tokens — still ~6/image. The cheap
combinations are `len>=3 + alnum` (136 tokens over 38 images) and `conf>=50 + len>=3 + alnum`
(**8 tokens over 7 images** — a 99.3% cut, reviewable in a minute).

### ⚠️ READ THIS BEFORE TOUCHING A2/A4 — the recall half was measured 2026-08-06, and it kills the premise

40 `ct_scout` series (Gradient's "patient info burned over anatomy" stratum) were taken through the
full chain — inputs → download → render → contrast check → seed → sweep — specifically to measure what
a gate *costs* on real text. Result: **526 tokens over 40 images, mean 13.2, median 10, 3 zeros.**

First, the renderer is exonerated on this stratum: `scripts/check_render_contrast.py` flagged **0/40**
frames as crushed. The 99.9%-binarized `ct_scout_00` seen earlier came from
`scan_text_poscontrol.py`'s own `apply_voi_lut` + min/max renderer, **not** from `ground_truth/render.py`,
whose D-10.1 path handles scouts correctly. Do not re-open that as a render bug.

Now normalize both sweeps per image (66 blank vs 40 scout) and take the ratio — scout tokens per blank
token. Above 1.0 means the gate favours real text; below means it favours noise:

```
gate                      blank/img   scout/img   ratio
ungated                      16.88       13.15     0.78
conf>=0  len>=3 alnum         2.06        1.90     0.92
conf>=30 len>=3 alnum         0.83        0.90     1.08
conf>=35 len>=1 alnum         4.45        4.10     0.92
conf>=40 len>=3 alnum         0.50        0.50     1.00
conf>=50 len>=1 alnum         1.88        2.00     1.06
conf>=50 len>=3 alnum         0.12        0.25     2.06   <- 8 vs 10 tokens; sampling noise
conf>=60 len>=1 alnum         1.21        1.00     0.82
conf>=70 len>=2 alnum         0.21        0.10     0.47
```

> ### 🛑 RETRACTED 2026-08-06 (same day) — READ THIS BEFORE THE TABLE ABOVE
>
> Arnav eyeballed ~30 of the 40 `renders/gt_ct_scout` frames: **not a single word on any of them.**
> `ct_scout` carries no burned-in text in this dataset. The 526 scout tokens are ~100% noise, so the
> comparison below is **noise against noise** and the flat ratio is exactly what two samples of one
> distribution look like. It is NOT evidence that Tesseract cannot separate text from anatomy — no
> text was present on either side. **The recall half of this measurement has not been done.**
>
> Still valid, because it needs no comparison stratum: **16.9 tokens/image on frames whose ground
> truth is exactly zero.** That false-positive rate stands, and so do the three suspected mechanisms
> (`--psm 11`, global Otsu, LANCZOS ringing), which were argued from algorithm behaviour rather than
> from this table. Redo the recall half on `ct_secondary_capture`; see plan.md §A2/§A2c.

**Essentially every gate sits between 0.78 and 1.10.** Whatever it filters, it removes noise and real
scout tokens at the same rate. At the high end the ratio drops *below* 1 — high-confidence detections
are marginally more common on the blank frames than on the text-bearing ones. This matches the
confidence medians: **26.0 on pure noise vs 28.5 on real burned-in text**, a 2.5-point gap.

**Conclusion: no confidence or length threshold separates real burned-in text from Tesseract's noise
on this data.** Consequences for the decisions below:

- **A2 and A4 have no evidence behind them.** Both pick a threshold the measurement says does not
  discriminate, then pay for it on every text-bearing stratum. Do not adopt either on the strength of
  the blank-stratum numbers alone — those numbers look persuasive only until the recall side is put
  next to them.
- **A1 and A3 are untouched.** A3 gets stronger: "keep everything, filter at view time" is the correct
  answer precisely when no fixed threshold is defensible.
- **B2 is directly undermined** for Tesseract — confidence carries almost no signal here. Whether
  docTR / PP-OCRv6 / EasyOCR behave the same way is unmeasured, and measuring it per engine is what
  B3 and B4 exist for.

### The one question that would overturn the above — ✅ **ANSWERED 2026-08-07, and it did overturn it**

*(Answer: `ct_scout` carries no text at all; the real recall measurement came from the ultrasound
pilot, and it killed the single-global-gate premise. See `### D-10d.A`. The framing below is kept
because it is the question that got asked correctly.)*

Those 526 scout tokens are an **unknown mix** of real text and noise; nothing in the sweep separates
them. If these 40 scouts happen to carry little burned-in text, then "no separation" would mean
"there was nothing to separate," and the finding would be about the sample rather than about
Tesseract.

**Resolve it before acting on any of the above:** open five or six `renders/gt_ct_scout/*.png` in a
standalone viewer and count real tokens by eye.

- ~10 genuine tokens/frame → the finding stands: Tesseract's noise drowns real text and gating
  cannot help.
- 1–2 genuine tokens/frame → the scouts are sparser than assumed; redo the recall half on
  `ct_secondary_capture`, which the poscontrol run showed is the text-densest stratum (173 regions on
  a single frame).

Also still unexplained: two `ct_axial` blank frames returned **265 and 145** tokens, ~37% of that
stratum's entire noise output. Whatever makes those two frames pathological is worth understanding
before a gate is chosen around a mean they distort.

Original framing, kept because it still governs which tokens a length floor costs: real clinical
tokens (patient IDs, accession numbers) are comfortably ≥3 chars, but short *KEEP* tokens — orientation
markers like `L`/`R`, `AP`, laterality letters — are exactly what a length floor deletes, and
`GTToken.label` is now binary PHI/KEEP, so those are in scope.

### D-10d.A — where seed filtering lives — ✅ **RESOLVED 2026-08-07: A3, filter at view time**

**Chosen: A3.** The seeder stays unfiltered (`seed_tesseract.py:281`, D-10c.4); 10d filters at
display time with per-stratum confidence defaults. Build it — see `## The view-time filter` above for
the spec and the table. **Do not re-argue A1/A2/A4** (kept below only as history).

**What decided it — the 2026-08-07 pilot.** 92 series were drawn from the strata nobody had looked
at; 86 were seeded and human-labelled. Three findings, in order of force:

1. **Ultrasound carries burned-in text and PHI on 48 of 48 images, all seven vendors.** The project's
   core premise, verified for the first time — and the strata that had no representation in any
   earlier sweep.
2. **A single global gate is wrong, and the earlier one was actively destructive.** `conf>=60, len>=2`
   — the gate the blank-stratum sweep above favoured — **deletes 29–47% of genuine burned-in text on
   ultrasound**, measured against human word counts:

   ```
   image (stratum)          human count   c10/l1   c20/l1   c30/l1   c60/l2
   1389495a (us_siemens)             17       25       23       17       10   -41%
   39e76f91 (us_philips)             34       40       37       33       24   -29%
   05c8e320 (us_ge)                 ~55       52       48       46       29   -47%
   ```

   Invention rates differ by **two orders of magnitude** across strata (`mg_2d` 99.6%, `us_ge` 24%),
   so no single threshold can serve both. Ultrasound overlay text is small and low-contrast, so
   Tesseract is genuinely uncertain about text that is really there and confidence stops separating
   real from invented. One frame (`31a50aa2`) shows the opposite failure: 247 tokens at `conf>=0`,
   still 242 at `conf>=70` — all-high-confidence invention, where no floor discriminates at all.
3. **Why nobody caught this sooner:** all seven `us_*` strata have **zero blank images**, so they have
   no invention floor and their image-level recall reads a saturated 100% at every gate. A gate tuned
   on `ct_axial` + `ct_secondary_capture` looked excellent and was silently deleting a third of the
   real text on the strata that actually matter.

**The three consequences that are now rules, not preferences:**

- **Filter at view time, not at seed time.** A token dropped by the seeder never appears in the UI and
  silently never reaches `gt.csv`. A token hidden by a view filter still exists on disk. Same reviewer
  workload; one is recoverable and one is not.
- **No length floor anywhere.** `9a9bd44b`'s entire real content is `R` and `L`.
- **Per-stratum, and loose where uncounted.** Tightening a stratum with no human word count behind it
  risks deleting real text — the expensive error.

Also settled by the pilot, for context: **`mg_tomo` has no text at all and the seeder correctly
returned exactly zero** — a third negative control alongside `ct_axial` and `ct_scout`, and (with
`mg_2d`'s 343 tokens/image on the same modality, run and config) the strongest evidence that invention
tracks pixel texture rather than a broken seeder.

#### Two questions still open with Cal — do NOT try to resolve them in code

Neither blocks 10d. Do not add a mechanism for either; just do not build anything that would have to
be undone when they land.

- **Do scanned documents belong in this benchmark?** `31a50aa2` and both `us_other` images are pages
  of sentences, not burned-in overlays — a different OCR task. They are scattered across strata, so
  they cannot be excluded by stratum. If they leave, they leave by `image_id`.
- **Ultrasound frame axis / `AmbiguousFrameAxisError` policy.** For ~30% of `us_*` series the
  manifest's frame count disagrees with what `render.py` counts in the tar, and `image_id` derives
  from `(sop_uid, frame_idx)` — so the resolution changes image identity. 4 of 92 pilot series hard
  -failed on it. If it changes, those images get new `image_id`s and their review JSONs are orphaned
  by name; that is Arnav's problem to sequence, not a case for you to handle defensively.

The three rejected options are kept below verbatim for the record.

#### (rejected options, for reference — A3 was chosen)

- **A1 — do not seed the blank control.** `ct_axial` GT is zero rows by definition, so every box is
  one you will reject. Zero labor, no chance of accepting a phantom. Cost: you *assert* emptiness
  rather than *review* it, and plan §2.3 requires a "blank" frame found to carry text to leave the
  control set — you'd rely entirely on `scan_text_ct_axial.py` (0/66 on `renders/gt`, detector
  validated) plus the human eyeball spot-check to catch that case.
- **A2 — re-seed with the gate baked into `row_filter`.** Uniform workflow across strata; a surviving
  token on a blank frame becomes a real flag. Costs: `row_filter` is recorded in provenance, so this
  is a **re-seed of everything, not an edit**; and the gate then applies to text-bearing strata too,
  where it costs recall — you would be tuning on blanks and paying on the strata that matter.
- **A3 — keep the seed unfiltered; filter at display time in 10d.** Seed stays maximal-recall and
  frozen; the UI gets a confidence/length threshold control, default high, sliding down when a frame
  looks suspiciously empty. This is insight #21 ("store raw, transform at compare time") applied to
  annotation, and the only option where changing your mind later costs nothing. Catch: it is a **10d
  requirement**, i.e. this decision changes what gets built.
- **A4 — change detection instead of filtering output.** `--psm 11` is Tesseract's most permissive
  mode and the 2× LANCZOS upscale amplifies texture that Otsu then reads as glyphs. `--psm 6` or a
  character whitelist cuts phantoms at the source. Also a re-seed — and psm 11 was presumably chosen
  *because* burned-in overlays are sparse scattered text, so it may cost real detections for exactly
  the reason it cuts noise.

A2 and A4 invalidate the current seeds; doing either **before** annotating anything is far cheaper
than after.

### D-10d.B — the confidence gate for the hallucination metric — ✅ **RESOLVED: B3 headline + B4 curve**

**Not a 10d change. Do not implement it here.** Recorded so the decision is not re-litigated, and
because it constrains 10d in one small way (below).

**Chosen:** per-engine operating points calibrated to **equal false-positive rate on the blank
control** (B3) are the headline ranking number; the **full threshold curve per engine** (B4) is
published alongside as the evidence. The curve is not extra work — sweeping each engine's threshold
against the blank control is how the equal-FPR point is found in the first place, so B4 is a byproduct
of B3. This satisfies the PM framing (a rankable number per engine) without hiding threshold
sensitivity.

**Rejected, with reasons:** B2 (one absolute floor) is unsound because confidence is not commensurable
across engines — Surya 2's is mean per-token *decode probability*, a different quantity from a
detection score, not a differently-calibrated version of one (CLAUDE.md §4). B1 (no gate) measures each
vendor's internal filtering policy as if it were reading quality. B3 alone was rejected only because
B4 is free given B3.

**Limitation that must be stated wherever the number is reported, not papered over:** 66 blank frames
is coarse. At the useful end of the sweep the calibration rests on single-digit token counts
(`conf>=50 + len>=3 + alnum` left 8 tokens over 7 images) — sampling noise, not signal. Either widen
the blank control from Gradient's canonical bucket, or publish a confidence interval on each operating
point. A precise-looking point estimate off 8 tokens is how a rigorous harness ends up wrong.

**Dependency worth naming:** B3 calibrates *on the blank control*, so the blank control must be
**reviewed**, not asserted. That is an independent argument against held-option A1.

**The one constraint on 10d — now live, because A3 ships a filter (updated 2026-08-07).** The UI's
view-time gate and the metric gate are **different instruments and must not be confused**: the UI gate
is Tesseract-seed confidence controlling what a human is shown, the metric gate is a per-engine
operating point controlling what gets scored. Two obligations follow:

1. Record the UI's effective gate in the review JSON in the same shape the metric gate uses —
   `{conf, len, alnum}` — so both are legible as the same kind of object.
2. Say in a comment at that field, and in your final report, that this value is **annotation
   provenance only**. It must never be read by 10e, never become a `gt.csv` column, and never be
   mistaken for a scoring threshold. A GT token's existence is a human decision; the gate only
   describes what that human was shown.

#### (options as originally framed, kept for the record)

- **B1 — no gate.** Score every token every engine emits. Honest about raw output, but penalizes
  engines that expose low-confidence candidates and rewards ones that filter internally — partly
  measuring vendor threshold policy rather than reading quality.
- **B2 — one absolute confidence floor for all engines.** Simple to state. But confidence is not
  comparable across engines — CLAUDE.md §4 flags this for Surya 2, whose confidence is *mean
  per-token decode probability*, not detection confidence. An absolute floor silently advantages
  whichever engine's scale runs hot.
- **B3 — per-engine operating point calibrated on the blank control.** Choose each engine's threshold
  as the one yielding an equal false-positive rate on the 66 blank frames, then compare reading
  quality at matched operating points. The apples-to-apples version; costs machinery and a clear
  write-up of how each point was chosen.
- **B4 — report the curve, not a point.** Tokens / false-redactions as a function of threshold, per
  engine. Makes threshold sensitivity visible instead of hidden; no single number for a ranking,
  which cuts against the PM framing of "rank engines by reading quality."

Whichever is chosen: the gate is **run metadata**, recorded and versioned like an engine version
(CLAUDE.md §9). Change it and prior results are not comparable.

---

## RESOLVED DECISION RECORD — build to exactly these (settled 2026-08-06, item 4 amended 2026-08-07)

Every open decision the Working-rhythm step 1 asked you to surface is answered here. **There is
nothing left for Arnav to decide before you write code.** Log each one as a comment where the code
governs it (Working rhythm step 3).

**1. Bind and port — hardcode `127.0.0.1:8765`. No flags, no env vars.**
No `--host`, no `--port`, no `HOST`/`PORT` env read. Both are module-level constants. The rationale
goes in a comment: the server needs no authentication *because* it cannot be reached from off-box, so
making the bind configurable would silently remove the only thing standing in for auth. 8765 over
8000/8080 to avoid colliding with a dev server. If it is ever busy, Arnav edits the constant. This
also makes required test 2 trivially strong — assert the constants and assert no `argparse` argument
or `os.environ` lookup can influence either.

**2. Keyboard shortcuts — minimal set, and a decision AUTO-ADVANCES.**
`A` accept · `E` enter edit mode · `D` defer · `←`/`→` prev/next image · `Del`/`Backspace` delete
selected box · `Esc` deselect. Deciding an image **writes the record and then advances** to the next
unreviewed image.

The obvious objection to auto-advance is that a mistaken keystroke commits and leaves the frame before
you notice. It is acceptable here **only because of the storage design already mandated**: one JSON per
image, written atomically, and re-opening an image shows the saved state. So `←` back to the previous
image plus a fresh decision fully rewrites that record — undo is free and needs no undo stack. Two
things must hold for that to be true, and both are on you:

- The write must complete **before** the advance (await the save response; do not fire-and-forget), or
  a fast `A A A` can advance past an unsaved image.
- `←`/`→` must navigate to *any* image including already-reviewed ones, not just unreviewed ones, or
  the undo path does not exist. The auto-advance target is the next **unreviewed** image; manual arrow
  navigation is unrestricted. Keep those two behaviours distinct.

Show the current image's saved state in the UI (e.g. `accepted` / `edited` / `deferred` / `unreviewed`)
so a re-opened image is visibly a re-open. Keyboard behaviour is **verified by Arnav by hand, not by a
test** — no browser automation, no screenshots (Rule zero). Say so in your report rather than implying
the shortcuts are test-covered.

**3. `deferred` carries a fixed enum reason PLUS an optional free-text note.**
Enum (exactly these four, no others, no free-form values in this field):
`unreadable` · `ambiguous_token` · `possible_non_blank_control` · `needs_cal`.

The enum exists because it is **PHI-free by construction**, so `--summary` can report it as counts —
and `possible_non_blank_control` is precisely the plan §2.3 signal ("a blank control frame found to
carry text must leave the control set") that otherwise lives only in Arnav's memory. Make it a required
field whenever `state == "deferred"`, and absent otherwise.

The optional free-text `note` is for Arnav's own later reference and is **PHI by assumption** — he may
well type a token into it. Therefore, as a hard rule enforced in code and asserted in tests:
`note` is never printed to stdout, never included in `--summary` output, never placed in a URL, header,
log line or error message, and never read by 10e. Only `state` and `defer_reason` are summary-visible.
Add a comment at the field's definition saying exactly that.

**4. View-time confidence filter, per stratum — D-10d.A = A3, RESOLVED. Build it.**
Full spec and the defaults table are in `## The view-time filter` near the top; the evidence is in
`### D-10d.A`. The four non-negotiables: hidden ⇒ **excluded from the record, not accepted**; **no
length floor at any stratum**; defaults live in **one config dict**, not inline; unlisted stratum ⇒
**`c0`, show everything**. The slider is confidence-only — do not add a length control.

**5. D-10.7 (GT-quality signal) — write `round` and `timestamp` now; the signal itself stays open.**
Every review JSON gets:
- `round`: int, `1` for a first review. A later self-agreement pass writes `round: 2` records to a
  separate directory (`ground_truth/review_r2/`) so round 1 is never overwritten — the whole point is
  comparing the two. Do **not** build the round-2 flow now; just make the field exist and default to 1,
  and make the output directory a module constant rather than a hardcoded string inline.
- `timestamp`: UTC ISO-8601, when the record was written.

Why now: with one annotator there is no inter-annotator agreement, so the candidate GT-quality signals
are (a) re-review a random ~10% after a gap and report self-agreement, (b) 10d's seed-quality stats
alone, or (c) a Cal spot-check. (a) and (c) both require knowing which round a record belongs to and
when it was made. Two fields plus one test now; a migration across 66+ hand-annotated JSONs later.
Arnav picks the signal after he has felt how long reviewing 66 images actually takes.

**Two constraints on these fields:**
- `timestamp` and `round` live in the review JSON **only**. They must never reach a `gt.csv` column —
  a timestamp in `gt.csv` makes its hash non-reproducible, which breaks the frozen-artifact guarantee.
  10a's frozen column spec already forbids it; do not add a column, and do not "helpfully" pass them
  through. Note it in the module docstring for whoever writes 10e.
- `timestamp` must not be used for ordering or dedup logic anywhere — D-1.1 stands, nothing here is
  order-dependent. It is provenance, not an index.

**6. `--summary` gains two blocks** on top of the three already specified (progress / seed quality /
box geometry):
- **Defer breakdown:** count per enum value. Never the `note`.
- **Round/timestamp coverage:** count of records per `round`, and the min/max timestamp as a date
  range only. This is what lets a future self-agreement pass verify it has a complete round-1 set
  before starting. No per-image timestamps.
- **Per-stratum breakdown, and gate coverage.** Every block above (progress, seed quality, geometry,
  defers) is **also reported per stratum**, not only pooled. Pooling twelve strata is the exact defect
  that made the `ct_scout` sweep meaningless and then hid the ultrasound gate failure for two days
  (CLAUDE.md §8) — a pooled-only summary here would repeat it a third time. Plus: how many tokens the
  gate hid per stratum, and the count of images reviewed at a **non-default** gate (a high count on one
  stratum is the signal that its default needs recalibrating).

**7. Two extra required tests** on top of the eleven already listed (number them 12 and 13):
- **12.** Round-trip the new fields: a `deferred` record with each of the four enum values reproduces
  its `state` + `defer_reason` on reopen; a record with a `note` reproduces the note in the reopened
  editor but the note string appears in neither `--summary` output nor captured stdout/stderr (assert
  on a fake note containing a `CMFN-00421`-style token). A `deferred` record missing `defer_reason` is
  rejected at write time.
- **13.** `round` defaults to 1 and `timestamp` parses as ISO-8601 UTC; per carry-over #3, assert on
  the artifact — read the written JSON back and check the values, do not assert that a setter was
  called.

## Done when
- All tests green, none skipped, `ruff check` clean.
- `ground_truth/review/` is gitignored AND hook-denied, with a test proving the deny.
- Decisions logged in comments; the geometry summary block exists and is tested.
- One commit naming the phase (e.g. `Phase 10d: localhost review UI + PHI rail extension for review
  JSONs`). Renders and review JSONs are never committed. Do not push.
- Your final report contains: the PHI-free summary field list, the runtime `netstat`/`ss` command for
  me, the managed-hook sync action item, an explicit statement that **you and every subagent** read no
  `.dcm`, render, review JSON, or `gt.csv`, a note that the keyboard/auto-advance behaviour is
  hand-verified rather than test-covered, the exact `--set` / `--groups` command line I should run for
  all 202 images, and the code-reviewer verdict with what you fixed vs. what you consciously left. If
  any step was skipped, say so loudly.

D-10.6 is settled: the purpose-built UI ships, Label Studio is not installed — do not reopen it. Do
not touch `harness/` scoring code, `iou_thr`, `normalize()`, or `gt.csv`.
