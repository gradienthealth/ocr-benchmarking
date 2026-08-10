"""Ground-truth construction (Phase 10). MOSTLY PHI-TOUCHING — see CLAUDE.md §0.

Only `gt_schema` (Phase 10a) is PHI-free and safe for Claude to read or import: it is
pure column-spec and validation logic with no pixel, token, or DICOM access. Every other
module that will land here (`render.py`, `seed_tesseract.py`, `review_gt.py`,
`build_gt.py`) handles PHI and is run by a human, never read back into Claude's context.

This file exists only to make the directory a real package (resolved 2026-08-05), so
`from ground_truth.gt_schema import load_gt` works identically under pytest, under an
editable install, and from any working directory.
"""
