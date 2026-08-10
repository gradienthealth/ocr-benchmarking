#!/usr/bin/env python3
"""PreToolUse hook: block PHI ingest, PHI egress, and destructive ops.

Claude Code is NOT covered by Anthropic's BAA (see CLAUDE.md §0), so any PHI it
reads is an out-of-BAA disclosure, and any command that could exfiltrate PHI or
destroy data is high-risk. This hook DENIES those actions before they run. It is
a backstop, deliberately pattern-based and conservative (errs toward blocking).

This same logic is also installed as a ROOT-OWNED managed hook
(/etc/claude-code/hooks/phi_guard.py) so it survives `clauded`
(--dangerously-skip-permissions) and cannot be disabled by the agent. Keep the
two in sync; they are intentionally redundant (defense in depth).

Protocol: tool call arrives as JSON on stdin. To block, print a JSON object with
permissionDecision "deny" and exit 0. To allow, exit 0 with no output.
"""
import json
import re
import sys

# ---- 1. PHI ingest: file contents Claude must never read ----
PHI_EXT = (
    ".dcm", ".dicom",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
    ".npy", ".npz",
)
# ADD-ONLY (Phase 10b): `*render_backmap*` is the image_id -> SOPInstanceUID/series_uid
# map written by ground_truth/render.py. It is PHI-adjacent (linkable identifiers) and must
# never reach Claude. `*render_inputs*` is the human-authored path+frame list, which also
# holds series UIDs. Both already fall under `ground_truth.*\.csv` at their default paths;
# these alternatives keep them READ-denied wherever they are put. Nothing above is loosened.
# KNOWN GAP (pre-existing, same for gt.csv): this list only gates the Read tool. `cat`/
# `grep` on these files via Bash is caught by §2a only if the path matches an image/DICOM
# pattern, which a .csv does not. Closing that means widening the Bash patterns for gt.csv
# too — a separate decision, not a Phase 10b change.
# ADD-ONLY (Phase 10c / D-10c.8): `ground_truth/seed/<image_id>.json` is the per-image
# Tesseract seed — it contains the read-back token text, i.e. the PHI itself, same class as
# gt.csv. The branch is deliberately `ground_truth/seed/.*` (trailing slash REQUIRED) and not
# `ground_truth/.*\.json`: the sibling `ground_truth/seed_summary.json` is a PHI-free count
# table Arnav must still be able to read, and a wider pattern would block it and defeat its
# purpose. `.*$` reaches end-of-string, so the branch satisfies the enclosing `(^|/)(...)$`.
# Nothing above is loosened.
# ADD-ONLY (Phase 10d): `ground_truth/review/<image_id>.json` is a human-confirmed GT review
# record — it holds the confirmed patient-ID/accession token text, i.e. the PHI itself, same
# class as gt.csv. The branch is `ground_truth/review.*/.*` (a `/` required after `review...`)
# rather than `ground_truth/.*\.json`: the wider form would also catch the PHI-free
# `ground_truth/seed_summary.json` sibling and defeat its purpose. This directory-scoped form
# matches `ground_truth/review/abc123.json` and a future self-agreement round
# `ground_truth/review_r2/x.json`, but NOT the source files `ground_truth/review_gt.py` or
# `ground_truth/review_ui.html` (neither has a `/` after `review`/`review_gt`/`review_ui`).
# Nothing above is loosened.
# ADD-ONLY (Phase 10d, found in review): the seed branch was `ground_truth/seed/.*`, which
# covered only the ORIGINAL seed dir. The per-stratum dirs added since — seed_v2, seed_pilot,
# seed_ct_sc, seed_ct_scout — hold the same read-back token text and `.gitignore:37` already
# treats them as PHI (`ground_truth/seed_*/`), but the Read hook did not. `seed(_[^/]*)?/`
# closes that gap. `[^/]*` cannot cross a path separator, so `ground_truth/seed_summary.json`
# (PHI-FREE per-image counts, which Arnav must still be able to read) stays allowed: there is
# no `/` after `seed_summary.json`. Nothing above is loosened.
#
# `.*\.part` (added 2026-08-09): an interrupted atomic write. `build_gt.write_gt()` stages the
# FULL gt.csv table at `ground_truth/gt.csv.part` before validating it, and 10d stages review
# records as `<image_id>.json.<pid>.<tid>.part`. Neither is caught by the branches above —
# `ground_truth.*\.csv` is `$`-anchored and `.part` is the final extension — so a crashed
# build leaves a readable full-PHI file behind. A `.part` file is by definition a temp
# artifact, so gating every one of them costs nothing.
PHI_NAME = re.compile(
    r"(^|/)(gt\.csv|ground_truth.*\.csv|.*raw_response.*|.*render_backmap.*|.*render_inputs.*"
    r"|ground_truth/seed(_[^/]*)?/.*|ground_truth/review.*/.*|.*\.part)$",
    re.I,
)

# ---- 2. Bash patterns, grouped by failure mode ----
# 2a. PHI -> stdout (returns to Claude's context)
PHI_READ = [
    (re.compile(r"\bg(cloud\s+storage|sutil)\s+cat\b", re.I),
     "streams object bytes (DICOM/tar/image) into Claude's context"),
    (re.compile(r"\btar\b[^|]*\s-[A-Za-z]*O", re.I),
     "`tar -O` extracts file contents to stdout (likely .dcm pixel/tag data)"),
    (re.compile(r"\b(cat|head|tail|less|more|xxd|od|hexdump|strings|base64)\b[^\n|]*\.(dcm|dicom|png|jpe?g|tiff?|bmp|npy|npz)\b", re.I),
     "dumping an image/DICOM file's bytes into context"),
    (re.compile(r"\b(dcmdump|gdcmdump|dcm2json|dcm2pnm|dcmj2pnm|dcmjson|dcm2xml)\b", re.I),
     "DICOM dump tools print identifying tags/pixels to stdout"),
    (re.compile(r"print\s*\(.*(pixel_array|PixelData|Dataset|dcmread|PatientName|PatientID|PatientBirthDate|token_text)", re.I | re.S),
     "printing DICOM pixels/tags/tokens routes PHI into context"),
    (re.compile(r"\b(imshow|Image\.open|display)\b[^\n]*\.(png|jpe?g|tiff?|bmp|dcm)\b", re.I),
     "opening/displaying a rendered image into context"),
    # ADD-ONLY (Phase 10c / D-10c.8): Bash-side companion to the PHI_NAME branch above. The
    # literal path segment `ground_truth/seed/` (trailing slash REQUIRED) keeps this narrow —
    # it cannot match `ground_truth/seed_summary.json` (PHI-free, must stay readable) nor
    # `ground_truth/seed_tesseract.py` (source). Closes for the seed files the KNOWN GAP noted
    # above for gt.csv; that gap is left as-is, nothing here is loosened.
    (re.compile(r"\b(cat|head|tail|less|more|jq|grep|strings|base64)\b[^\n|]*ground_truth/seed/", re.I),
     "dumping a Tesseract seed file (contains read-back PHI token text) into context"),
    # ADD-ONLY (Phase 10d): Bash-side companion to the PHI_NAME branch above. The literal path
    # segment `ground_truth/review/` (trailing slash REQUIRED) keeps this narrow — it cannot
    # match `ground_truth/review_gt.py` or `ground_truth/review_ui.html` (source, must stay
    # readable). Nothing above is loosened.
    (re.compile(r"\b(cat|head|tail|less|more|jq|grep|strings|base64)\b[^\n|]*ground_truth/review[^/\s]*/", re.I),
     "dumping a review record (contains human-confirmed PHI token text) into context"),
    # ADD-ONLY (Phase 10d, found in review): Bash companion for the per-stratum seed dirs
    # (seed_v2, seed_pilot, ...) that the `ground_truth/seed/` literal above never covered.
    # `[^/\s]*` cannot cross a separator, so `ground_truth/seed_summary.json` (PHI-free) is not
    # matched — it has no trailing `/`. Nothing above is loosened.
    (re.compile(r"\b(cat|head|tail|less|more|jq|grep|strings|base64)\b[^\n|]*ground_truth/seed_[^/\s]*/", re.I),
     "dumping a per-stratum Tesseract seed file (read-back PHI token text) into context"),
]
# 2b. Network egress (could send PHI off the machine / to a non-BAA endpoint)
EGRESS = [
    (re.compile(r"\b(curl|wget|telnet|ftp|sftp|scp|rsync)\b", re.I),
     "network egress tool — could exfiltrate PHI to a non-BAA endpoint"),
    (re.compile(r"\b(nc|ncat|netcat|socat)\b", re.I),
     "raw socket tool — could exfiltrate PHI"),
    (re.compile(r">\s*/dev/tcp/", re.I),
     "bash /dev/tcp network write — could exfiltrate PHI"),
    (re.compile(r"\bg(cloud\s+storage|sutil)\s+cp\b.*series", re.I),
     "copying PHI series objects to another location"),
    (re.compile(r"requests\.(post|put|patch)\s*\(|urllib|http\.client|smtplib", re.I),
     "in-script HTTP/email call — could exfiltrate PHI"),
]
# 2c. Destructive / hard-to-reverse
DESTRUCTIVE = [
    (re.compile(r"\bgit\s+push\b", re.I),
     "git push — could publish PHI to a remote; do this deliberately, not via the agent"),
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.I),
     "recursive force delete — confirm the target by hand"),
    (re.compile(r"\b(dd|mkfs\S*|shred)\b", re.I),
     "low-level disk/destructive command"),
]
# ---- 3. Credential reads ----
CRED_PATH = re.compile(r"(^|/)(\.env(\.|$)|\.aws/|\.ssh/|\.config/gcloud/|credentials($|\b)|.*\.pem$|.*\.key$)", re.I)

DENY_TEMPLATE = (
    "BLOCKED by PHI/safety hook. {reason}.\n"
    "This repo handles PHI and Claude Code is not under a BAA (see CLAUDE.md §0). "
    "Instead, have a human run the sensitive step, or write a script that emits only "
    "PHI-free aggregates (counts/metrics/shapes/hashes) to a file. If this is a false "
    "positive, ask the user to confirm before overriding."
)


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": DENY_TEMPLATE.format(reason=reason),
        }
    }))
    sys.exit(0)


def check(data: dict) -> None:
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}

    if tool in ("Read", "NotebookRead"):
        path = str(ti.get("file_path") or ti.get("notebook_path") or "")
        if path.lower().endswith(PHI_EXT) or PHI_NAME.search(path):
            deny(f"reading a PHI file is not allowed: {path}")
        if CRED_PATH.search(path):
            deny(f"reading a credential/secret file is not allowed: {path}")

    elif tool == "Bash":
        cmd = str(ti.get("command", ""))
        for group in (PHI_READ, EGRESS, DESTRUCTIVE):
            for pat, reason in group:
                if pat.search(cmd):
                    deny(reason)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # never break the tool on a parse error
    check(data)
    sys.exit(0)


if __name__ == "__main__":
    main()
