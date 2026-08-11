#!/usr/bin/env bash
# Preflight for the overnight 13i scored run. Run this WHILE AWAKE, before launching
# run_overnight_13i.sh. ~11 minutes. 🔴 STAGE 3 IS PHI-TOUCHING (it opens renders).
#
#   bash experiments/preflight_13i.sh
#
# WHY THIS EXISTS: scripts/run_experiment.py is ~57 KB written in a single session and has
# never been executed against real frames. Launching a 3.5-hour unattended job from a
# never-run script is how you wake up to a stack trace at 7am with nothing on disk. The
# three stages below escalate cost: hashes (seconds) -> synthetic (~15 s) -> the fastest
# real engine (~10 min). A failure surfaces while you can still do something about it.
#
# Stage 3 deliberately runs docTR alone. At ~2.8 s/image it is the cheapest possible proof
# that the real-frame path works end to end: renders open, the backmap joins, the matcher
# runs, aggregate() writes. If stage 3 passes, the only thing the overnight run adds is
# more minutes of the same code path.
#
# stdout is PHI-free (counts, rates, hashes, identities) but is redirected to a log anyway,
# per CLAUDE.md §7 — Claude reads the summary artifacts, not this stream.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=experiments/preflight_13i.log
: > "$LOG"

# Timestamped out-dirs rather than `rm -rf` on a fixed path. `--out` must be new or empty, and
# a destructive delete inside an unattended overnight chain is not worth the convenience.
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== STAGE 1/3 — hashes only, no models, no pixels ===" | tee -a "$LOG"
.venv/bin/python scripts/run_experiment.py --verify-only --text-presence ground_truth/text_presence_v2.csv 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== STAGE 2/3 — synthetic dry run, no PHI, no network (~15 s) ===" | tee -a "$LOG"
.venv/bin/python scripts/run_experiment.py --synthetic --all-arms --out "results/preflight_synthetic_${STAMP}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== STAGE 3/3 — 🔴 REAL FRAMES, docTR only, 199 images (~10 min) ===" | tee -a "$LOG"
.venv/bin/python scripts/run_experiment.py --arm doctr:stock --renders renders/v2 --backmap ground_truth/render_backmap_v2.csv --manifest manifest.csv --text-presence ground_truth/text_presence_v2.csv --out "results/preflight_doctr_${STAMP}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== PREFLIGHT PASSED — safe to launch run_overnight_13i.sh ===" | tee -a "$LOG"
