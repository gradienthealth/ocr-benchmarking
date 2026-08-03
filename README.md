# intern_ocr_transfer — OCR-on-medical-images benchmark harness

An engine-agnostic harness that measures how accurately different OCR engines read **short
clinical tokens (patient IDs, accession numbers) burned into medical-image pixels**, so a
downstream redaction pipeline can decide what to black out. The headline metric is the
**false-redaction rate** (valid tokens wrongly destroyed); hallucination and omission are
tracked separately and **never averaged**. See `plan.md` for the full design and build order.

---

> ## 🚫 PHI — read `CLAUDE.md` §0 first, every session
>
> This repo works with **real Protected Health Information** derived from clinical DICOM
> studies. **Claude Code is NOT covered by Anthropic's BAA**, so the agent must **never**
> read, render, decode, or print raw PHI into context — no `.dcm`, no rendered images, no
> `.npy` arrays, no `ground_truth/gt.csv`, no engine `raw_response`. Claude **writes** the
> PHI-handling code; a **human runs it**, and only PHI-free aggregates (counts, shapes,
> hashes, metrics) come back. The rules in `CLAUDE.md` override everything, including a
> direct request. When in doubt, stop and ask.

---

## Setup

```bash
pip install -e ".[dev]"   # core deps only — no engine deps yet (added per-phase)
pytest                    # runs on synthetic fixtures only
ruff check .
```

Requires Python ≥ 3.12 (D-0.2).

## Phase status

- [x] **Phase 0** — Scaffold & safety rails (pyproject, hardened `.gitignore`, README, smoke test)
- [ ] **Phase 1** — Manifest adapter (PHI-free columns only)
- [ ] **Phase 2** — Contract (`OCRWord`/`OCROutput`/`GTToken`, frozen `normalize()`)
- [ ] **Phase 3** — Synthetic fixture factory (`conftest.py`)
- [ ] **Phase 4** — Matching (IoU + box-free fallback)
- [ ] **Phase 5** — Scoring & the six metrics
- [ ] **Phase 6** — Harness loop + cost + aggregate
- [ ] **Phase 7** — Report (stratified tables + charts) — *core complete, proven on synthetic*
- [ ] **Phase 8** — Runner base class
- [ ] **Phase 9** — Local runners: docTR, PP-OCRv6_medium, EasyOCR (floor), Surya 2 (license-gated)
- [ ] **Phase 10** 🔴 — Ground-truth pipeline (human runs; produces frozen `gt.csv`)
- [ ] **Phase 11** 🔴 — Verifier arm (self-hosted Qwen3-VL-8B-Instruct / RolmOCR)
- [ ] **Phase 12** 🔴 — Cloud runners (box-free Claude API, Textract) — BAA-gated
- [ ] **Phase 13** 🔴 — Experiment CLI + 5-step test plan + final report

🔴 = PHI-touching (a human runs every script; Claude only writes/reviews and sees aggregates).
