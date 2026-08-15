#!/usr/bin/env bash
# Assemble every PHI-free result the overnight run produced into ONE file. 🟢 PHI-FREE.
#
#   bash experiments/morning_summary.sh
#
# Runs as Job 4 of run_overnight_13i.sh, and is safe to re-run by hand any time.
#
# WHY: the 10am readout has a hard 9:21am workstation shutdown in front of it, and Arnav may
# not wake at exactly 7. A late wake must not be spent working out which of several results
# directories completed. This collapses that to `cat experiments/MORNING_SUMMARY.md`.
#
# Everything it copies is a PHI-free aggregate — TABLES.md holds counts, rates, latencies,
# config hashes and version strings; identity.json holds hashes. No token text, no crops, no
# predicted strings (scripts/run_experiment.py never writes them anywhere). Claude may read
# this file.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/MORNING_SUMMARY.md
: > "$OUT"

{
    echo "# Morning summary — assembled $(date -u +%FT%TZ)"
    echo ""
    echo "Everything below is PHI-free aggregate output. Source of truth for slides 15-17."
    echo "Deck must be off this workstation before its 09:21 local hard shutdown."
    echo ""
    echo "## Run status — read this first"
    echo ""
} >> "$OUT"

# Newest first, so a relaunched night puts its latest attempt at the top.
found_any=0
# `found_any` counts a dir ONLY if it has run_status.json. run_experiment.py creates its --out
# directory BEFORE the hash gate and the render checks, so any early abort leaves an EMPTY dir
# behind. Counting those made the "NOTHING COMPLETED -> use the fallback" branch unreachable
# after a total failure: the summary would list bare dirs as UNREADABLE and never print the
# pointer to the fallback prompt, which is the one instruction a 7am reader would need most.
for d in $(ls -1dt results/gt_v1_* 2>/dev/null); do
    [ -d "$d" ] || continue
    [ -f "$d/run_status.json" ] || { echo "- \`$d\` — **EMPTY, aborted before it started** (no run_status.json)" >> "$OUT"; continue; }
    found_any=1
    status=$(python3 -c "import json,sys; print(json.load(open('$d/run_status.json'))['status'])" 2>/dev/null || echo "UNREADABLE")
    tables="MISSING"
    [ -f "$d/TABLES.md" ] && tables="present"
    echo "- \`$d\` — status **${status}**, TABLES.md ${tables}" >> "$OUT"
done
for d in results/preflight_doctr_*; do
    [ -d "$d" ] || continue
    [ -f "$d/run_status.json" ] || continue
    found_any=1
    status=$(python3 -c "import json,sys; print(json.load(open('$d/run_status.json'))['status'])" 2>/dev/null || echo "UNREADABLE")
    echo "- \`$d\` — status **${status}** (preflight docTR-only insurance run)" >> "$OUT"
done
if [ "$found_any" -eq 0 ]; then
    echo "- **NOTHING COMPLETED.** No results dir exists. Go to plan.md appendix A.4," >> "$OUT"
    echo "  the 'no numbers' fallback prompt, and build the deck from the dev-slice" >> "$OUT"
    echo "  findings in A.5. Do not invent numbers." >> "$OUT"
fi

{
    echo ""
    echo "A status of COMPLETE means its tables were written. INCOMPLETE means the run died"
    echo "partway and that directory has NO table — run_experiment.py writes all-or-nothing."
    echo ""
} >> "$OUT"

# --- the tables, newest run first ---------------------------------------------------------
for d in $(ls -1dt results/gt_v1_* results/preflight_doctr_* 2>/dev/null); do
    [ -f "$d/TABLES.md" ] || continue
    {
        echo ""
        echo "---"
        echo ""
        echo "# FROM: $d"
        echo ""
        cat "$d/TABLES.md"
    } >> "$OUT"
done

# --- the dev-slice tuning footnotes -------------------------------------------------------
for d in experiments/sweep_dev_v1_doctr experiments/sweep_dev_v1_easyocr experiments/sweep_dev_v1_paddle; do
    [ -f "$d/sweep_report.txt" ] || continue
    {
        echo ""
        echo "---"
        echo ""
        echo "# DEV-SLICE TUNING FOOTNOTE (22 images, NOT the scored set): $d"
        echo ""
        echo '```'
        cat "$d/sweep_report.txt"
        echo '```'
    } >> "$OUT"
done

echo ""
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
