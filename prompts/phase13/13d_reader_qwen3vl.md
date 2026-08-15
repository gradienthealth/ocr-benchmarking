# 13d — Arm C reader: Qwen3-VL-4B (self-hosted) 🟡 build PHI-free, run is PHI-touching

> **Prerequisites: 13a + 13b merged; 13c is a useful reference but not required.**
> This arm doubles as the evidence for **D-11.1** (which VLM becomes the Phase-11 verifier), so its
> results feed two decisions.
>
> **Weights stay in-environment. No API call, no egress, ever.** Arnav runs it on real crops.

---

## Goal

Qwen3-VL-4B-Instruct reading GT crops through the 13b path, self-hosted, with its hallucination floor
measured before any accuracy number is quoted.

## Context

- `harness/reading.py` — the reader interface and pinned crop preprocessing (13b).
- plan.md PHASE 11, **D-11.1 (reopened)** — the A/B this feeds, and why the 8B pick was never
  measured: Qwen2.5-VL-7B scored 0.962/0.943 precision/recall on synthetic imprinted text and
  collapsed to **0.719/0.858 on real burned-in text** (arXiv 2511.02014).
- CLAUDE.md §4 — Qwen3-VL is ✅ **only if self-hosted**.
- `ground_truth/text_presence_*.csv` — which images are blank, for the negative control. The blank
  control set is **64**, not 66 (two `ct_axial` frames turned out to have text).

## Constraints

- **Self-hosted only.** Weights load from local/HF cache and run in-process. If any code path could
  reach a hosted inference endpoint, delete it. No `requests`, no API client, no remote base URL.
- **Deterministic decode**: temperature 0, fixed seed, pinned revision. Record the exact model
  revision/commit hash in run metadata (rule #9) — a silent model rev is an invalidated result.
- **The prompt given to the VLM is part of the config identity.** Two runs with different prompts are
  different arms (13a). Keep the prompt in one constant, versioned, not inlined at the call site.
- **Do NOT let it see the whole frame.** It reads one crop at a time — that is the arm's definition,
  and it is also minimum-necessary.
- **Abstention beats invention.** Design the prompt so an unreadable crop returns a defined empty
  marker rather than a guess, and test that path. No untrained VLM abstains reliably
  (arXiv 2511.19806) — measure it, don't assume it.
- Confidence, if extracted, is a **decode logprob** — a different family from a detection score.
  Label it as such; never let it share a threshold with docTR's or PP-OCR's confidence.
- Tests are synthetic-only. Real crops are Arnav's to run.

## Steps

1. Enter **Plan Mode**. The plan must enumerate: model loading and revision pinning, the prompt
   constant, the empty/abstain contract, dtype and device selection for CPU, the files to create, and
   the test list.
2. Implement the reader + synthetic tests (known token, determinism across two runs, abstain on a
   blank crop, charset sanity).
3. Write a short **CPU timing script** that reports mean/median/p95 seconds per crop over synthetic
   crops. No public figure exists for this model on CPU; measure it before anyone plans around it.
4. Write the **negative-control procedure** for Arnav to run: the reader over crops from the 64 blank
   control images, emitting only counts (how many crops produced non-empty text). Any non-empty
   output there is invention.

## Output format

Plan first. Then the diff, the synthetic test output, and the CPU timing numbers from the synthetic
run. Finish with the exact commands Arnav runs for the negative control.

## Done when

- `pytest` green; `ruff check .` clean.
- Synthetic CPU timing reported (mean/median/p95 s per crop).
- A grep of the new code shows no HTTP client, no API key handling, no remote endpoint.
- The negative-control command is written down and emits counts only — never token text.

## Finally

Spawn one fresh-context subagent to review. Scope: any possible network egress, any path where the
model sees more than one crop, whether the abstain contract actually holds, and whether the decode
logprob could be mistaken for a detection confidence downstream. Correctness and constraint
violations only.
