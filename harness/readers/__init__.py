"""Phase 13c — reader implementations for the 13b reading arm. 🟢 PHI-free package init.

The `Reader` ABC, the pinned crop preprocessing and the scoring path all live in
`harness/reading.py`; this package holds only the per-model adapters, one module each,
exactly as `harness/runners/` holds one module per engine. The split is the same fairness
rule: **model-specific knowledge lives in the adapter, never in the harness** — the crop a
reader is handed and the way its string is scored are identical across readers, so a
difference in the numbers is a difference in reading quality (CLAUDE.md, engine-agnostic
rule).

Deliberately empty of imports. Importing a reader pulls in its model library (torch, a
recognizer, sometimes a weight download), so `harness.readers` re-exporting them would make
every importer of the package pay for every model. Import the module you want:

    from harness.readers.read_svtrv2 import SVTRv2Reader

**Readers are not engines.** A reader is scored under an oracle detector (it is given the GT
box) and its results are tagged `arm="reader"` by `run_reading`; they belong in their own
table and are never blended into the end-to-end ranking (plan.md, "Step 7 in detail").
"""
