#!/usr/bin/env bash
# Phase 13h / D-13.4 — the stock-vs-tuned threshold sweep on `dev_v1`. 🔴 PHI-TOUCHING.
#
#   bash experiments/run_sweep_dev_v1.sh            # measure only, nothing frozen
#   bash experiments/run_sweep_dev_v1.sh --freeze   # AND write the winners
#
# Deliberately NOT frozen by default. The report carries both ranking metrics for every
# trial, so the winner can be re-selected from sweep_report.json without re-running any
# engine — look at the numbers first, decide the metric, then freeze.
#
# Runs 13 configurations (docTR 5, PP-OCRv6 6, EasyOCR 2) over 22 images on CPU. Expect
# tens of minutes. stdout is PHI-free: counts, rates, latencies, config hashes.

set -euo pipefail
cd "$(dirname "$0")/.."

exec .venv/bin/python experiments/sweep_stock_vs_tuned.py \
    --renders renders/dev_v1 \
    --gt ground_truth/dev_v1.csv \
    --backmap ground_truth/render_backmap_dev_v1.csv \
    --manifest manifest.csv \
    --out experiments/sweep_dev_v1 \
    "$@"
