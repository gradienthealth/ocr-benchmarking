#!/usr/bin/env bash
# Phase 13h / D-13.4 — build the tuning slice `dev_v1`. 🔴 PHI-TOUCHING (Arnav runs it).
#
# A script rather than a pasted one-liner: the build command is long enough that terminal
# paste keeps splitting it mid-flag, which fails loudly here but silently truncated the
# dev-slice DRAW earlier (only 1 of 4 --exclude files applied, and the summary still looked
# plausible). Same reasoning as scripts/download_series_tars.py.
#
#   bash experiments/build_dev_v1.sh
#
# stdout is build_gt's PHI-free summary: counts, hashes, per-stratum tables. No token text.

set -euo pipefail
cd "$(dirname "$0")/.."

RENDERS="renders/dev_v1"
SEEDS="ground_truth/seed_dev_v1"
REVIEW="ground_truth/review_dev_v1"

# The two out-of-scope document pages (experiments/dev_v1_dropped.md). Guarded rather than
# assumed: they were ACCEPTED unedited, so leaving either in puts a page of unverified
# Tesseract output into the artifact the thresholds get tuned against.
DROPPED=(75ea0cd7 cf23f1c1)
fail=0
for id in "${DROPPED[@]}"; do
    if grep -q "^${id}" "${RENDERS}/render_manifest.csv"; then
        echo "ABORT: ${id} is still in ${RENDERS}/render_manifest.csv — it was dropped as an" >&2
        echo "       out-of-scope document page. Remove its row before building." >&2
        fail=1
    fi
    if [ -f "${REVIEW}/${id}.json" ]; then
        echo "ABORT: ${REVIEW}/${id}.json still exists — delete it, or build_gt's orphan" >&2
        echo "       gate will reject the whole build anyway." >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || exit 1

echo "guards passed: both dropped images are out of the set"
echo

exec .venv/bin/python ground_truth/build_gt.py \
    --set "${RENDERS}:${SEEDS}" \
    --groups ground_truth/seed_groups_dev_v1.csv \
    --backmap ground_truth/render_backmap_dev_v1.csv \
    --manifest manifest.csv \
    --drawn ground_truth/render_inputs_dev_v1.csv \
    --out ground_truth/dev_v1.csv \
    --review-dir "${REVIEW}" \
    --text-presence-name text_presence_dev_v1.csv
