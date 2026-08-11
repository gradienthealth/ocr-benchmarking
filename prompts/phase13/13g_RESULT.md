# 13g RESULT — Gemini 2.5 Pro reader, BAA-gated

Branch `13g`, worktree `.worktrees/13g` (own real venv — nothing installed into the shared
benchmark venv). Base: `a6b9c95` (13b merged, so the reader path exists).

---

## 1. What shipped

| File | What it is |
| --- | --- |
| `harness/readers/read_gemini.py` | `GeminiReader(Reader)` + the gate. New. |
| `tests/test_read_gemini.py` | 38 tests, all offline/synthetic. The gate test is first. |
| `tests/test_readers.py` | registers this arm in the shared scaffold, gated on `GEMINI_LIVE=1`. |
| `harness/cost.py` | `+ gemini` pricing rule, `+ gemini_in`/`gemini_out` rates. |
| `experiments/bench_reader_synthetic.py` | Exact-match bench over fixed synthetic crops. New. |
| `pyproject.toml` | `+ [gemini]` extra (`google-genai==2.17.0`); E501 ignore for the bench. |

**It is a Reader, not a Runner.** It subclasses `harness.reading.Reader`, takes a crop and
returns a string — the same shape as SVTRv2, docTR-PARSeq and Qwen3-VL. It was originally
written at `harness/runners/read_gemini.py`, which straddled both conventions (a `read_`
name in the `runners` directory); it now sits with the other three. **Its numbers belong in
the Arm C reader table, never in the end-to-end ranking** — every reader in that arm is
handed oracle boxes, so it is not comparable to an engine that had to find the text itself.

**Registered in the shared scaffold under `GEMINI_LIVE=1`.** Qwen3-VL's opt-in is about
cost; this one is about egress — reading a crop makes a network call to a BAA-covered
endpoint, so registering it unconditionally would mean anybody's `pytest` silently billed a
Google project and sent pixels out of the environment. An opted-in reader that fails to
import is a failure, not a silent skip. The instance is wired to the digest of exactly the
one crop the shared tests use, so it can transmit that crop and nothing else.

**One shared claim is exempted, in writing.** The scaffold's version-sourcing check asserts
`version == importlib.import_module(version_source).__version__`. No installed library
carries a hosted model's version. `version_source = "google.genai"` would pass the test and
be wrong — the SDK version says nothing about what the model returns — so the reader is
named in `VERSION_SOURCE_EXEMPT` with the reason, mirroring `runners/base.py`'s precedent.
An *accidental* `version_source=None` still fails. The revision is pinned from the response
instead (§5).

**The gate, in one paragraph.** Every reader is constructed with an explicit `source=` and
there is no default. `CropSource.REAL` requires `OCR_BAA_CLEARED_GEMINI=1`, compared with
`==` (not truthiness — `"0"` and `"no"` must not open a gate), checked at construction *and*
immediately before every single call, above the transport, so an injected transport, a
subclass, or a reader that outlives its clearance cannot route around it.
`CropSource.SYNTHETIC` needs no flag, but does not take the caller's word either: the reader
is handed the sha256 digests of the crops it may transmit and hashes every crop against that
set, so a real render fails **on its pixels**, not on anyone's honesty.

**Residual hole, stated rather than papered over:** someone who hashes real crops and passes
those digests in gets a live call without the flag. No code path in this repo produces such a
set. The guarantee is "a real render cannot reach the network by mistake," not "by deliberate
act."

**Two more properties the prompt asked a grep for:** the module opens no credential file
(env only) and persists no response body — no `raw_response`, no disk write, no log line, no
print. Both are asserted against the module source in the test file, because the failure mode
is a debug line someone adds in six months, not a branch a behavioural test would reach.

---

## 2. Test + lint

After merging main (13c + 13d), in the **repo-root** venv — the only one with openocr and
docTR installed, which `tests/test_readers.py::test_every_reader_is_registered` requires:

```
617 passed, 3 skipped, 1 xfailed
tests/test_read_gemini.py: 38 passed   (offline; no network, no credentials)
```

The 13g work was developed in a **worktree-local** venv (`.worktrees/13g/.venv`, a real
directory, not a symlink) because it installs `google-genai` — CLAUDE.md forbids putting
model dependencies in the shared benchmark venv, since that silently changes the environment
docTR/PP-OCRv6/EasyOCR are measured in. `google-genai` is **not** in the repo-root venv, and
nothing was installed there. The suite passes in both; the worktree venv reports one failure,
`test_every_reader_is_registered`, which is that test correctly noticing openocr and docTR
are absent there.

**Lint: 27 errors repo-wide, identical to main's pre-existing count** (16 UP007, 4 E501,
3 UP037, 2 UP035, 1 I001, 1 F841). This branch adds none. Worth a separate cleanup pass; not
folded into a BAA-gated change.

---

## 3. Synthetic exact-match comparison

18 crops from 6 fixed scenes (3 clean + 3 `hard=True`), all fake `CMFN`/`GRDN`/`ACC` tokens,
cut by the pinned `crop_for_reading()` so every arm sees identical pixels. 14 of the 18 are
8-character tokens — the length the ~59% projection is quoted against.

Both arms measured 2026-08-11. Gemini ran live against Vertex on `gradient-health-central`.

| Arm | Exact match | 8-char only | Misread | Empty | s/crop | $/crop |
| --- | --- | --- | --- | --- | --- | --- |
| docTR `crnn_vgg16_bn` (local reference) | **17/18 = 94.4%** | 13/14 = 92.9% | 1 | 0 | 0.071 (CPU) | $0 |
| **gemini-2.5-pro** (Vertex) | **17/18 = 94.4%** | 13/14 = 92.9% | 1 | 0 | **7.71** | $0.0028 |

**They tie on accuracy and miss the same token.** The one failure in the set is `GRDN0001`
for both, and the two error shapes are different in a way worth keeping:

| Arm | Expected | Read |
| --- | --- | --- |
| docTR | `GRDN0001` | `GRDNO001` — digit zero read as letter O |
| gemini-2.5-pro | `GRDN0001` | `GRDN00001` — an extra zero inserted |

A run of repeated zeros immediately after letters defeats both, one by character confusion
and one by miscounting a repetition. docTR's `0`→`O` is the classic identifier-corruption
class and the more dangerous of the two for an allowlist match; Gemini's is a length error a
format check would catch. Neither is a hallucination: both produced a plausible near-miss on
a token that is genuinely there.

**The ~59% projection was wrong on this material, and that is not a vindication.** Fixture
glyphs are Pillow-rendered, clean, evenly spaced and noise-free. Burned-in overlay text is
none of those. What this table establishes is that the arm is correctly wired and reads easy
text as well as the incumbent — not that it will hold up on `gt_v1`.

**The separation is not accuracy, it is everything else.** 7.71 s/crop against 0.071 — **108x
slower** — and $0.0028/crop against $0, on an arm that also ships every crop out of the
environment. On the 2351-token `gt_v1` set that is roughly 5 hours and ~$6.50 versus about
3 minutes and nothing. Accuracy parity on easy text does not survive that comparison unless
the real-frame numbers separate sharply in Gemini's favour.

**A false start worth recording** (it is the reason the first live run read 16.7%): thinking
tokens count against `max_output_tokens` on 2.5 Pro and thinking cannot be disabled, so the
original 64-token ceiling with a 128-token thinking budget truncated 5 of 18 crops to empty
before a character was emitted. Those scored as omissions and looked exactly like a model
that could not read. The reader now records the API's `finish_reason` per call — the
corrected run reports `FinishReason.STOP` 18/18 — so a truncation can never again be
mistaken for a reading failure. **The config change means the two runs are different arms:**
`config_hash` moved `28d8a5ed2e8f` → `153b822c7ded`, which is the D-13.5 guard working.

**Cost estimator is known-inaccurate for this arm.** Measured 1,676 input tokens per crop;
`harness/cost.py`'s `_gemini_tokens` predicts ~303 for a 48px crop — off by 5.5x. Nothing
downstream is wrong, because the arm is priced on `usage_summary()["measured_cost_usd"]`, but
the estimate should not be quoted for Gemini. Deliberately not retuned to match a single run.

**Version identity is weaker than the design wanted.** Vertex reports `model_version` as
`gemini-2.5-pro` — the alias itself, not a concrete revision. The response-side pin works and
would abort on a mid-run change, but it cannot manufacture precision Google does not publish.
This arm's identity is therefore "whatever `gemini-2.5-pro` was on 2026-08-11," and the run
date belongs in the final report next to the version string. Say so plainly rather than
implying a revision was pinned.

**Reproduce** (PHI-free, no clearance needed):

```
GOOGLE_CLOUD_PROJECT=gradient-health-central GOOGLE_CLOUD_LOCATION=us-central1 .venv/bin/python experiments/bench_reader_synthetic.py --reader gemini --show-predictions --out results/reader_synth.json
```

---

## 4. D-12.1 pre-flight checklist — Gemini 2.5 Pro

Confirm **in writing**, per service, before `OCR_BAA_CLEARED_GEMINI=1` is ever set.
Status as of 2026-08-11.

- [x] **A signed Google Cloud BAA covering Gradient exists.** Confirmed in writing by
      Gradient, 2026-08-11: *"we need a BAA in place with any 3rd party in order to send
      PHI. We have one with Google of course, and OpenAI. I'm working on one with Anthropic.
      … Model usage within Vertex AI is acceptable (e.g., Claude via Vertex AI)."*
      *Still worth recording for the file:* a link to that message and where the
      countersigned BAA lives.

- [x] **The product actually called is covered.** Settled by the same statement — model usage
      **within Vertex AI** is acceptable, which is the surface this arm calls. Note what it
      does **not** cover: the **Gemini Developer API (AI Studio, `GEMINI_API_KEY`)** is a
      separate product outside that sentence, and it is one `genai.Client(api_key=…)` away.
      The code enforces the distinction — `_live_transport` builds
      `genai.Client(vertexai=True, project=…, location=…)` and refuses to start without a
      project and location, so an API key alone cannot reach it. Keep the run on
      `gcloud auth application-default login` against a Gradient GCP project.

- [ ] **Retention / logging configured.** Vertex generative APIs: confirm prompt logging and
      any request caching are OFF for the project, and record the data-residency region
      chosen in `GOOGLE_CLOUD_LOCATION`. Retention config is part of the BAA obligation, not
      an optimization.

- [x] **The model id is servable on Vertex.** Settled 2026-08-11 by running the synthetic
      bench against `gradient-health-central`: `gemini-2.5-pro-preview-06-05` **404s** — the
      dated preview revisions were retired at GA — and `gcloud ai model-garden models list`
      shows the only 2.5 Pro entry is `google/gemini-2.5-pro@default`. **The pin therefore
      moved from the request to the response** (see §5). The 404 also proved out auth,
      project, region, API enablement and the service account's permissions, since the
      request reached Vertex and was rejected on the model id alone.

- [ ] **Named confirmer + date.** One line: who confirmed the two boxes above, and when.
      Cal is the point of contact for anything that needs Gradient-side sign-off.

- [ ] **The human runs it, outside Claude's sandbox** (D-12.3, CLAUDE.md §0). The Google BAA
      covers Google. It does not cover Claude Code, which remains outside Anthropic's BAA —
      so the render, the crop, and the read-back string must never pass through this agent's
      context regardless of how well cleared the Gemini call is. `run_reading()` writes
      PHI-free aggregates to a file; that file is what comes back here.

**Nothing in this diff sets the variable, and nothing reads it from a committed file.** It
lives in one person's shell, after the boxes above are ticked.

---

## 5. Amendment 2026-08-11 — the revision is pinned from the response, not the request

The original design rejected any model id without a dated revision, on the grounds that
`gemini-2.5-pro` is a moving target. Vertex does not offer the alternative: on
`gradient-health-central`, `gemini-2.5-pro-preview-06-05` returns `404 NOT_FOUND` and
`gcloud ai model-garden models list` shows exactly one 2.5 Pro entry,
`google/gemini-2.5-pro@default`. Dated preview revisions are retired at GA.

Rejecting the only servable id would have made the arm unrunnable, and accepting it silently
would have put rows into the report stamped "some 2.5 Pro." So the rule moved rather than
bending:

- The **request** carries the alias (`requested_model`), which is all Vertex accepts.
- The **response** carries the revision that actually answered, and that becomes the arm's
  `version` — pinned on the first reply.
- `version` **raises** if read before any call has resolved it. Every caller of `version` is
  stamping an identity onto a result row, and a placeholder there is exactly the ambiguity
  rule #9 exists to forbid. `run_reading()` reads it after the first crop, so the ordinary
  path never sees the raise.
- A later reply naming a **different** revision aborts the run (`ServedVersionMismatch`). We
  cannot stop Google re-pointing an alias mid-run; we can refuse to average the two halves
  and call it a measurement.
- An alias plus a reply that reports **no** revision is fatal. An unattributable number is
  not a measurement.
- If Google ever publishes dated revisions again, passing one still pins at request time,
  no call required — that path is kept and tested.
- `config()` keeps the **requested** id, not the served revision. The served one already
  lands in `version`, which `aggregate()` keys on separately; putting it in the config hash
  too would split one config into two the moment Google re-points the alias, for a reason
  that is not a config change.

Tests: 38 in `tests/test_read_gemini.py` (was 36), suite 546 passed / 28 skipped, ruff clean
on the files touched.

---

## 6. Running it on real `gt_v1` frames — what a human does, and what does not exist yet

**There is no driver.** `run_reading()` has exactly two callers in the repo: its own tests
and nothing else. The experiment CLI that wires renders + the frozen `gt.csv` + a reader into
it is **13i**, which is not built. Setting the gate variable today gets a cleared reader with
nothing to feed it. `13i_experiment_cli.md` now carries the constraints this arm needs — its
own `--arm gemini:real` selector that "all arms" does not pick up, an abort if the variable
or the Vertex project is missing, the negative control before any accuracy number, measured
rather than estimated cost, and no retry layer.

When 13i exists, clearing the gate is three steps, in this order:

1. Finish the two open boxes in §4 (retention/logging config, named confirmer).
2. Prove the plumbing on synthetic crops first — it needs no clearance and costs ~$0.05:

```
GOOGLE_CLOUD_PROJECT=gradient-health-central GOOGLE_CLOUD_LOCATION=us-central1 .venv/bin/python experiments/bench_reader_synthetic.py --reader gemini --show-predictions --out results/reader_synth.json
```

3. Then, and only then, in a shell that a human typed this into:

```
OCR_BAA_CLEARED_GEMINI=1 GOOGLE_CLOUD_PROJECT=gradient-health-central GOOGLE_CLOUD_LOCATION=us-central1 .venv/bin/python scripts/run_experiment.py --arm gemini:real --gt ground_truth/gt.csv
```

That last command is **written against 13i's not-yet-existing surface** and its flags will
need checking against what 13i actually builds. Arnav runs it, outside Claude's sandbox
(D-12.3): the Google BAA covers Google, not Claude Code.

Budget for the real pass, extrapolated from the measured synthetic run: 2351 tokens at
7.71 s/crop and $0.0028/crop is roughly **5 hours and $6.50**, against about 3 minutes and
$0 for a local reader.

---

## 7. Outstanding

- **Two D-12.1 boxes are open** (§4): Vertex prompt-logging/caching confirmed off for the
  project, and a named confirmer + date on the covered-product statement.
- **No driver for real frames** — blocked on 13i, above.
- **Negative control not run for this arm.** plan.md requires every generative arm to clear
  the blank-frame hallucination floor *before* its accuracy numbers are believed, and this
  arm is generative. `experiments/reader_negative_control.py` (13d) exists and is the right
  vehicle; it has not been pointed at this reader. **Its 94.4% should not be quoted in the
  final report until it has.**
- **Cost estimator known-inaccurate here** — measured 1676 input tokens/crop against ~303
  estimated. Priced on measured usage, so nothing downstream is wrong; deliberately not
  retuned to match one run.
- **Version identity is an alias**, not a revision (§5). The run date is part of it.
- **Synthetic numbers only.** n=18 clean fixture crops. Not a `gt_v1` result.
