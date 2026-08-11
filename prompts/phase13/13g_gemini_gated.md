# 13g — Gemini 2.5 Pro reader 🔴 BAA-GATED — synthetic only until cleared

> **This is a Phase 12 arm, not part of the local bake-off.** It is cloud egress: every crop sent
> leaves the environment. It cannot run on a real frame until the BAA line item is confirmed
> **in writing**.
>
> **Prerequisite: 13b merged** (it rides the same reader path). Do this one last — it is gated on a
> human decision, not on code.
>
> **AMENDED 2026-08-11 — the human decision was made.** Gradient's Google BAA is signed and model
> usage **within Vertex AI** is acceptable (confirmed in writing). This prompt's scope is unchanged
> and still ends at synthetic crops — but the arm is now cleared for real `gt_v1` frames, which
> **13i** wires up. See `13g_RESULT.md` §4 for the completed D-12.1 checklist and the two remaining
> mechanical items. The AI Studio `GEMINI_API_KEY` surface is still uncovered; Vertex only.

---

## Goal

A Gemini 2.5 Pro reader that is fully exercised on **synthetic** `CMFN`-style tokens, and that is
structurally incapable of sending a real render anywhere until the BAA gate is explicitly opened.

## Context

- CLAUDE.md §4 — the vendor/BAA matrix. Google Cloud products are covered **only where the product
  itself is on the BAA-covered-products list**; "Google has a BAA" is not sufficient for a specific
  model on Vertex.
- CLAUDE.md §0 and §6.2 — never send PHI to a non-BAA service.
- plan.md PHASE 12 — the D-12.1 checklist that must be confirmed per service before any real PHI.
- `harness/reading.py` — the reader interface (13b).
- Expectation to test against, not to assume: **~59% projected exact match on 8 characters.** It is
  expected to lose to the local readers; the point is to have the number, not to promote it.

## Constraints — the gate is the feature

- **Hard-fail on real data by default.** The reader must refuse to run unless an explicit,
  human-set environment variable (e.g. `OCR_BAA_CLEARED_GEMINI=1`) is present. Absent that, any
  non-synthetic input path raises immediately with a message naming CLAUDE.md §4. Do not make the
  flag default-on, do not read it from a committed file, and do not let a test set it globally.
- **Synthetic-only path must be the easy one.** Running against `tests/` fixtures needs no flag.
- **Never log or persist the response body** beyond the scored string — a cloud response holds
  read-back token text (CLAUDE.md §3). No `raw_response` written to disk.
- Credentials come from the environment only. **Never** read `~/.config/gcloud`, never hardcode a
  key, never print one.
- Deterministic settings (temperature 0) and the exact model version string recorded in run metadata
  (rule #9). `gemini-2.5-pro` without a pinned revision is not a version.
- Cost: record per-call token/character counts so `harness/cost.py` can price the arm honestly.
- Its confidence, if any, is a **decode-side quantity** — never share a threshold with a detection
  confidence.

## Steps

1. Implement the reader + the gate, with synthetic tests only.
2. Add a test asserting that a non-synthetic path **raises** when the env var is absent. That test is
   the point of this prompt — write it first.
3. Run the synthetic `CMFN`-style benchmark and report exact-match rate versus the local readers'
   numbers on the same fixtures.
4. Write the **D-12.1 pre-flight checklist** for this service into the session output: exactly what
   Arnav must have in writing (Gemini-on-Vertex named on the covered-products list, zero-retention
   configuration if applicable, and who confirmed it), before the env var is ever set.

## Output format

Diff, then test output, then the synthetic exact-match comparison table, then the D-12.1 checklist.

## Done when

- `pytest` green, including the gate-raises test; `ruff check .` clean.
- A grep shows no credential file read, no response-body persistence.
- The synthetic comparison table exists.
- Nothing in the diff can send a real render anywhere without a human setting the env var.

## Finally

Spawn one fresh-context subagent with a single question: is there any code path — including tests,
fixtures, error handlers, and retries — by which a non-synthetic image could reach the network
without `OCR_BAA_CLEARED_GEMINI` being set by a human? Report only paths it can actually construct.
