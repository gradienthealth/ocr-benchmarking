#!/usr/bin/env bash
# 2026-08-11 morning chain: wait out the in-flight PP-OCRv6 run, then run easyocr:tuned.
# 🔴 PHI-TOUCHING — Arnav runs this. Launch it in a SECOND terminal; do not touch the first.
#
# WHY THIS EXISTS. The overnight chain died at 09:21 UTC — NOT the scheduled shutdown (that was
# 16:21 UTC and never fired) but the 1 h CLOUD_WORKSTATIONS_IDLE_TIMEOUT, ~1 h after the laptop
# stopped holding the connection open. Jobs 1-2 (docTR, EasyOCR stock) are on disk. Job 3
# (PP-OCRv6 stock, THE CONTENDER) was 35 min in and wrote nothing — run_experiment.py is
# all-or-nothing. Job 4 (easyocr:tuned, Cal's question) never started.
#
# Arnav relaunched PP-OCRv6 by hand at 13:40 UTC in a foreground terminal, WITHOUT a `timeout`
# wrapper. This script does not interfere with it. It waits for that process to exit, then runs
# easyocr:tuned on the same cores it just vacated. That is the whole point: nobody has to wake up
# in an hour to type the second command.
#
# ⚠️ THIS CANNOT SURVIVE AN IDLE STOP, AND NEITHER CAN nohup OR setsid. The idle timeout stops the
# whole VM; every process on it dies, this waiter included. setsid protects the run from the SHELL
# hanging up, which is a different failure. If the connection to this workstation drops while you
# sleep, you get ~1 h of compute from that moment and PP-OCRv6 needs more. Laptop on power, lid
# open, workstation tab open. That single fact is what cost last night.
#
# The wall-clock shutdown is not the binding constraint any more: it reset on the 13:32 UTC reboot
# to 2026-08-12T01:31Z (18:31 local). The 10am readout is safe on this box.
#
# ⚠️ IT KILLS A HUNG PP-OCRv6 AT THE CUTOFF. Cutoff 15:55 UTC = 08:55 local, RAISED from 15:40
# once the run's real progress became measurable (see MEASURED PROGRESS below): 15:40 was only
# ~30 min past the measured ETA, close enough that a slightly slow finish would have killed the
# contender nine-tenths of the way through. The board is worth more than easyocr:tuned, so the
# margin goes to PP-OCRv6. Past 15:55 it has lost the readout either way,
# and a killed run_experiment.py writes no table whether it is killed now or at 10am — so nothing
# is destroyed that was not already gone, and easyocr:tuned's ~30 min is still recoverable before
# the talk. SIGINT not SIGTERM, so its own failure path runs and marks run_status.json.
# To disable and wait forever instead:  MORNING_NO_KILL=1 bash experiments/run_morning_13i.sh
#
# TIMING. Both mean-weighted estimates in plan.md A.2 came in ~3x pessimistic on real frames
# (docTR 13 min est -> 5m11s actual; EasyOCR 85 min est -> 28m14s actual). Applying that ratio to
# PP-OCRv6's ~3 h gives ~45-90 min, and its killed attempt ran 35 min without finishing, so it is
# not shorter. Inference from two data points, NOT a measurement. Launched 13:40 UTC, so expect it
# 14:25-15:10 UTC (07:25-08:10 local), then easyocr:tuned lands ~07:55-08:40 local.
#
# MEASURED PROGRESS — how to know where a run actually is, since run_experiment.py prints no
# per-image progress. load_scored_images (run_experiment.py:331) hands the harness the scored set
# in SORTED image_id order, and Paddle prints "Resized image size (WxH) exceeds max_side_limit of
# 4000" exactly once per image whose long side is over 4000. There are 17 such images in the 199,
# at sorted positions 4, 12, 19, 25, 35, 68, 74, 101, 102, 110, 111, 135, 151, 166, 172, 175, 198.
# So counting those warnings locates the run to the image, and their WxH sequence confirms the
# mapping rather than assuming it. Recompute with a script that reads ImageRef.w/h only — sizes
# are the non-PHI signal CLAUDE.md §0 sanctions, and no pixel is touched.
#
# Fitting the two observations (last night: position 74 in ~35 min before the idle stop; this
# morning: position 135 in 62 min) gives ~12 s per ordinary frame and ~3.1 min per mammogram, and
# the two runs agree on throughput to within 3%. Hence the 15:09 UTC ETA the cutoff is set around.
# Reuse this trick on any engine that logs one line per large image.

set -uo pipefail
cd "$(dirname "$0")/.."

RENDERS=renders/v2
BACKMAP=ground_truth/render_backmap_v2.csv
PRESENCE=ground_truth/text_presence_v2.csv
COMMON="--renders $RENDERS --backmap $BACKMAP --manifest manifest.csv --text-presence $PRESENCE"

# Absolute epoch, not an offset: this script may sit in `sleep` for over an hour and an offset
# computed at launch would drift with it. 15:55 UTC.
CUTOFF_EPOCH=$(date -u -d '2026-08-11T15:55:00Z' +%s)

# --- guard: the dev sweep must not be eating cores alongside a scored run ------------------
# p95 latency is a REPORTED metric on slide 15. Shared cores produce inflated, unreportable
# timings (13d measured per-crop cost drifting 26 -> 30 s under load). The [s] bracket is not a
# typo — it stops the pattern matching this script's own shell.
if pgrep -f '[s]weep_stock_vs_tuned' > /dev/null; then
    echo "ABORT: a dev sweep is running. Its numbers and this run's would both be wrong:"
    echo "         pgrep -af '[s]weep_stock_vs_tuned'"
    exit 1
fi

echo "=== morning chain started $(date -u +%FT%TZ) ==="

# --- phase 1: wait out whatever scored run is in flight ------------------------------------
# If none is running (the hand-launched PP-OCRv6 already died, or was never started), fall
# through immediately and run PP-OCRv6 here instead. Same arm, its own timestamped --out, so a
# relaunch can never collide with or clobber a completed directory.
if pgrep -f '[r]un_experiment.py' > /dev/null; then
    echo "waiting on the in-flight scored run:"
    pgrep -af '[r]un_experiment.py'
    while pgrep -f '[r]un_experiment.py' > /dev/null; do
        if [ -z "${MORNING_NO_KILL:-}" ] && [ "$(date -u +%s)" -ge "$CUTOFF_EPOCH" ]; then
            echo "CUTOFF $(date -u +%FT%TZ) — the in-flight run has had its 2 h and is hung."
            echo "Sending SIGINT so it writes run_status.json, then moving to easyocr:tuned."
            pkill -INT -f '[r]un_experiment.py'
            sleep 60
            pkill -KILL -f '[r]un_experiment.py' 2>/dev/null
            break
        fi
        sleep 60
    done
    echo "cores free at $(date -u +%FT%TZ)"
    bash experiments/morning_summary.sh > /dev/null 2>&1
    echo "MORNING_SUMMARY.md refreshed — check whether PP-OCRv6 says COMPLETE or RUNNING."
else
    echo "no scored run in flight — running PP-OCRv6_medium stock here first (limit 2h)."
    echo "=== PP-OCRv6_medium stock, frozen gt_v1 (THE CONTENDER) started $(date -u +%FT%TZ) ==="
    # -s INT so run_experiment.py's failure path actually runs: SIGTERM terminates CPython without
    # raising, so its `except BaseException` never sees it and run_status.json is never updated.
    # Verified: `timeout -s INT` still exits 124, so the check below is unaffected.
    timeout -s INT -k 2m 2h .venv/bin/python scripts/run_experiment.py --arm pp-ocrv6_medium:stock $COMMON --out "results/gt_v1_paddle_$(date +%Y%m%d_%H%M)"
    rc=$?
    [ "$rc" -eq 124 ] && echo "PP-OCRv6 HIT ITS 2h TIMEOUT — it hung rather than failed. No table."
    echo "PP-OCRv6 exit=${rc} at $(date -u +%FT%TZ)"
    bash experiments/morning_summary.sh > /dev/null 2>&1
fi

# --- phase 2: easyocr:tuned, its own invocation and its own --out --------------------------
# Separate invocation on purpose: run_experiment.py writes tables only after EVERY selected arm
# finishes, so bundling would let this arm's failure destroy the contender's table.
# This arm builds even though tuned_configs.json is empty — EasyOCR's stock/tuned pair is fixed by
# definition (library 0.7/0.4 vs the 0.2/0.2 this project shipped), not selected by a sweep.
# doctr:tuned and pp-ocrv6_medium:tuned still raise ArmNotFrozenError. Nothing is frozen.
echo ""
echo "=== easyocr:tuned = the shipped 0.2/0.2, frozen gt_v1 started $(date -u +%FT%TZ)  (limit 90m) ==="
timeout -s INT -k 2m 90m .venv/bin/python scripts/run_experiment.py --arm easyocr:tuned $COMMON --out "results/gt_v1_easyocr_tuned_$(date +%Y%m%d_%H%M)"
rc=$?
[ "$rc" -eq 124 ] && echo "easyocr:tuned HIT ITS 90m TIMEOUT — it hung rather than failed. No table."
echo "easyocr:tuned exit=${rc} at $(date -u +%FT%TZ)"
bash experiments/morning_summary.sh > /dev/null 2>&1

echo ""
echo "================ MORNING CHAIN DONE $(date -u +%FT%TZ) ================"
echo "Exit 0 = fine.  124 = hung and was killed at its limit.  anything else = failed."
echo "easyocr:tuned exit=${rc}"
echo ""
echo "A job with no exit code at all means the VM stopped under it — its results dir will say"
echo "RUNNING with no TABLES.md. That is the idle timeout, not a bug. Nothing to salvage."
echo ""
echo "ON WAKING: bash experiments/morning_summary.sh; cat experiments/MORNING_SUMMARY.md"
echo ""
