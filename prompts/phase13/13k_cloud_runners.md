# 13k — Cloud runners (Anthropic API box-free, AWS Textract) 🔴 BAA-GATED

> **Was `prompts/phase12/12_cloud_runners.md`; renumbered into the Phase-13 sequence 2026-08-10.**
> The phase numbers were never chronological — plan.md PHASE 13's test plan invokes this as its
> **step 4**, so it belongs after the verifier arm (13j) and before the report (13l).
> plan.md's own **PHASE 12** section is unchanged and remains the spec this implements.
>
> **Only run this if a cloud arm is actually wanted.** Per plan.md PHASE 13 step 4, cloud is a
> fallback **only if self-host loses**. If docTR / PP-OCRv6 / the readers are good enough, skip it.
> **Conditional by design — the expected outcome is that this is never built.**
>
> **Nothing here sends a real frame anywhere.** This session builds runners + mocked tests. Live calls
> are Arnav's, outside Claude's sandbox, only after D-12.1 is confirmed in writing per service.

---

## Goal

Two cloud runners behind an off-by-default BAA flag, proven against **mocked** responses, plus the
written D-12.1 checklist that must be satisfied before either is ever pointed at real data.

## Context

- plan.md **PHASE 12** — D-12.1 (BAA + config per service), D-12.2 (`raw_response` handling),
  D-12.3 (the human runs live calls, not Claude).
- CLAUDE.md §4 — the vendor matrix. **Anthropic's API is BAA-coverable; Claude Code is not.**
  OpenAI requires BAA **and** zero-data-retention endpoints. **Mistral OCR stays excluded — no BAA.**
- `harness/runners/base.py` — the `Runner` contract; `run_claude.py` sets `OCROutput.box_free=True`.
- `harness/cost.py` — cost model; Claude box-free is `(w·h)/750`, Textract is flat per image.
- 13g (`prompts/phase13/13g_gemini_gated.md`) — the gate pattern is already established there; reuse
  the same shape rather than inventing a second one.

## Constraints

- **Off by default, hard-fail without the flag.** Each runner refuses to touch non-synthetic input
  unless a human-set environment variable is present. Never default-on, never read the flag from a
  committed file, never let a test set it globally.
- **`raw_response` is never logged, printed, or written in the clear** — it contains the read-back
  PHI. Store hashed/redacted only (D-12.2, CLAUDE.md §8).
- **CI and tests use mocked responses only.** No live call in any test, ever.
- Credentials come from the environment. Never read `~/.aws`, `~/.config/gcloud`, or any key file;
  never print a key.
- **Mistral OCR is excluded** — do not add it, do not re-litigate.
- Box-free arms (Claude) score through the existing box-free path; do not invent a second scoring
  route. Textract returns boxes + confidence → convert **inside its runner**.
- Record exact model/API version strings in run metadata (rule #9).

## Steps

1. Build `runners/run_claude.py` (Anthropic **API**, box-free) with mocked-response tests.
2. Build `runners/run_textract.py` (boxes + per-word confidence → `OCROutput`) with mocked-response
   tests, including a coordinate-conversion test on a synthetic image.
3. Write the **D-12.1 checklist** as a markdown block: per service, exactly what must be confirmed in
   writing, by whom, and where the confirmation is recorded — before the flag is ever set.
4. Add a test that each runner **raises** on non-synthetic input when the flag is absent.

## Output format

One runner at a time: diff, then its test output. Then the D-12.1 checklist block.

## Done when

- `pytest` green including both gate-raises tests; `ruff check .` clean.
- A grep shows no credential-file read and no `raw_response` written in the clear.
- The D-12.1 checklist exists and names both services explicitly.

## Finally

Spawn one fresh-context subagent with one question: is there any path — tests, fixtures, retries,
error handlers — by which a non-synthetic image reaches a network call without a human setting the
flag? Report only paths it can actually construct.
