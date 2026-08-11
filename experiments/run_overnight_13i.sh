#!/usr/bin/env bash
# The overnight run for the 2026-08-11 10am readout. 🔴 PHI-TOUCHING — Arnav runs this.
#
# Do not launch this directly — use the one-line command in plan.md appendix A.0, which chains
# the preflight in front of it, appends to experiments/tonight.log, and always runs the morning
# summary afterwards even if this script dies.
#
# ONE ENGINE PER INVOCATION. This is the load-bearing decision, and it was changed at 05:55
# after two audits. run_experiment.py writes its tables only after EVERY selected arm in an
# invocation finishes — a run that dies partway leaves run_status.json INCOMPLETE and emits NO
# table at all. PP-OCRv6 is simultaneously the longest arm (~2.8-3.3 h over 199 images) and the
# only one with no completed measurement anywhere in this repo, i.e. the most likely to hang.
# Bundling it with docTR and EasyOCR for the sake of one pre-merged Table A would let the
# riskiest arm destroy the two safe ones. So each engine runs alone and Table A gets assembled
# from three files at 7am — MORNING_SUMMARY.md already concatenates them in order.
#
# ORDER IS BY (value / risk), cheapest-and-safest first, so the essentials are on disk early:
#
#   JOB 1  docTR stock          ~10 min   the incumbent. On disk by ~00:25.
#   JOB 2  EasyOCR stock        ~85 min   the floor. Table A has two of three rows by ~01:50.
#   JOB 3  PP-OCRv6 stock       ~3 h      THE CONTENDER, and the at-risk arm. Board complete ~04:50.
#   JOB 4  easyocr:tuned        ~75 min   the question Cal will ask. ~06:05.
#   JOB 5  Paddle dev sweep     ~2 h      tuning-symmetry footnote. May not finish — that is fine.
#   JOB 6  Arm C reader svtrv2   ~30 min   Table C bonus. May not finish. Also fine.
#   JOB 7  Arm C reader parseq   ~30 min   Table C bonus. May not finish. Also fine.
#
# Timings are OFFSETS from launch, not clock times — see plan.md A.2, which also carries the
# latest-safe-launch table (board needs ~4h40, board + Cal's question ~6h, hard stop 09:21).
#
# Jobs 5-7 are deliberately last and deliberately expendable. Everything the presentation needs
# is done by ~T+6h even on the pessimistic estimate. If you wake and jobs 5-7 are still running,
# that is the plan working, not failing — read the summary and let them be, or stop them.
#
# Job 5 has NO CHECKPOINTING: its sweep writes a report only after all configs finish, so exit
# 124 there means zero output, not partial. Nothing to salvage; do not go looking.
#
# EVERY JOB IS WRAPPED IN `timeout`. run_experiment.py has no internal deadline and no signal
# handling, so one wedged CPU inference would hang forever, write nothing, and burn the night
# silently. Exit 124 means "hung", not "failed" — the messages say so explicitly.
#
# THE SUMMARY IS REBUILT AFTER EVERY JOB, not just at the end, so the readout is readable the
# moment it exists rather than after the expendable work finishes.
#
# NOT IN SCOPE TONIGHT, deliberately:
#   * doctr:tuned / pp-ocrv6_medium:tuned — need a dev-slice --freeze first, and
#     experiments/tuned_configs.json is still EMPTY, so they raise ArmNotFrozenError. Note that
#     both sweep_report.txt headers claim the winners are "FROZEN"; that text is wrong, nothing
#     was frozen (--freeze was never passed). Do not quote it.
#   * reader:qwen3vl — ~26 s/CROP over 2,351 crops is ~17 h, and `transformers` is not in the
#     root venv. The only genuinely unaffordable reader; the other two are Jobs 6 and 7.
#   * gemini:real — BAA-gated cloud egress, needs a human to set OCR_BAA_CLEARED_GEMINI=1 after
#     the D-12.1 checklist. Not something to do at 4am.

set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M)

# Derived, never hand-counted. The banner previously hardcoded its total, and renumbering the
# jobs silently desynced plan.md's "wait for JOB 1/N" babysit check — a 20-minute wait that
# could never match. plan.md now greps the total-agnostic prefix "=== JOB 1/" as well.
TOTAL=7

RENDERS=renders/v2
BACKMAP=ground_truth/render_backmap_v2.csv
PRESENCE=ground_truth/text_presence_v2.csv
COMMON="--renders $RENDERS --backmap $BACKMAP --manifest manifest.csv --text-presence $PRESENCE"

# --- guard: nothing else may be eating the cores -----------------------------------------
# p95 latency is a REPORTED metric on slide 15. A scored run sharing cores with a dev sweep
# produces inflated, unreportable timings — 13d measured per-crop cost drifting 26 -> 30 s under
# load. Refuse rather than quietly corrupt a number that goes on a slide.
# The [s] bracket is not a typo: a plain `pgrep -f sweep_stock_vs_tuned.py` also matches any
# shell whose own command line contains that string, which makes the guard fire on itself.
if pgrep -f '[s]weep_stock_vs_tuned' > /dev/null; then
    echo "ABORT: a dev sweep is still running. Its latency numbers and this run's would"
    echo "       both be wrong. Wait for it, or stop it deliberately, then re-launch:"
    echo "         pgrep -af '[s]weep_stock_vs_tuned'"
    exit 1
fi

# run_job <n> <label> <timeout> <outdir-or-> <command...>
run_job() {
    local n="$1" label="$2" limit="$3"; shift 3
    echo ""
    echo "=== JOB ${n}/${TOTAL} — ${label} ==="
    echo "started $(date -u +%FT%TZ)  (limit ${limit})"
    # -s INT, not the default TERM, and not -k alone. Proven at 07:55: SIGTERM and SIGKILL
    # terminate CPython without raising, so run_experiment.py's `except BaseException` never
    # runs, ledger.fail never fires, and a timed-out run keeps the "RUNNING" status it wrote at
    # start -- indistinguishable from a VM stop. SIGINT raises KeyboardInterrupt, which that
    # except path DOES catch, so the run records INCOMPLETE. -k 2m is the backstop for a wedge
    # inside a C extension, where the Python-level handler cannot run until control returns.
    # Verified: `timeout -s INT` still exits 124, so the detection below is unaffected.
    timeout -s INT -k 2m "$limit" "$@"
    local rc=$?
    [ "$rc" -eq 124 ] && echo "JOB ${n}/${TOTAL} HIT ITS ${limit} TIMEOUT — it hung rather than failed."
    echo "JOB ${n} exit=${rc} at $(date -u +%FT%TZ)"
    bash experiments/morning_summary.sh > /dev/null 2>&1
    echo "JOB ${n}: MORNING_SUMMARY.md refreshed."
    eval "JOB${n}=${rc}"
}

run_job 1 "docTR stock, frozen gt_v1 (Table A row 1 + Table B)" 60m \
    .venv/bin/python scripts/run_experiment.py --arm doctr:stock $COMMON --out "results/gt_v1_doctr_${STAMP}"

# 2h not 3h: the four non-expendable jobs must fit the wall clock even in the worst case. At
# 60m+3h+5h+3h the budget was 12h against a ~9h window, so Jobs 2 and 3 both timing out would
# have killed Job 4 — "Cal's question", which is not expendable — at the hard shutdown. EasyOCR's
# own mean-weighted estimate is 85 min, so 2h is still generous.
run_job 2 "EasyOCR stock = library 0.7/0.4, frozen gt_v1 (Table A row 2, the floor)" 2h \
    .venv/bin/python scripts/run_experiment.py --arm easyocr:stock $COMMON --out "results/gt_v1_easyocr_${STAMP}"

run_job 3 "PP-OCRv6_medium stock, frozen gt_v1 (Table A row 3 — THE CONTENDER)" 5h \
    .venv/bin/python scripts/run_experiment.py --arm pp-ocrv6_medium:stock $COMMON --out "results/gt_v1_paddle_${STAMP}"

run_job 4 "easyocr:tuned = the shipped 0.2/0.2, frozen gt_v1 (records config_id 'thr0.2')" 3h \
    .venv/bin/python scripts/run_experiment.py --arm easyocr:tuned $COMMON --out "results/gt_v1_easyocr_tuned_${STAMP}"

run_job 5 "PP-OCRv6 stock-vs-tuned on dev_v1 (footnote, measure only, expendable)" 3h \
    .venv/bin/python experiments/sweep_stock_vs_tuned.py --renders renders/dev_v1 --gt ground_truth/dev_v1.csv --backmap ground_truth/render_backmap_dev_v1.csv --manifest manifest.csv --out experiments/sweep_dev_v1_paddle --engine pp-ocrv6_medium

# Split into two invocations. Both audits flagged the bundling, and the second proved why it
# matters: run_real re-raises on the first arm failure and RunLedger.finish() writes tables
# all-or-nothing, so a timeout during arm 2 would leave a FULLY COMPLETED svtrv2 arm with no
# TABLES.md at all — its numbers stranded in arms/reader_svtrv2/aggregate.json, which
# morning_summary.sh does not read. Expendable work still should not destroy finished work.
run_job 6 "Arm C reader: SVTRv2 on oracle GT boxes (Table C, expendable)" 2h \
    .venv/bin/python scripts/run_experiment.py --arm reader:svtrv2 $COMMON --out "results/gt_v1_reader_svtrv2_${STAMP}"

run_job 7 "Arm C reader: docTR-parseq on oracle GT boxes (Table C, expendable)" 2h \
    .venv/bin/python scripts/run_experiment.py --arm reader:doctr-parseq $COMMON --out "results/gt_v1_reader_parseq_${STAMP}"

echo ""
echo "================ OVERNIGHT SUMMARY ================"
echo "Exit 0 = fine.  124 = hung and was killed at its limit.  anything else = failed."
echo ""
echo "JOB 1  docTR stock          exit=${JOB1:-?}   THE BOARD"
echo "JOB 2  EasyOCR stock        exit=${JOB2:-?}   THE BOARD"
echo "JOB 3  PP-OCRv6 stock       exit=${JOB3:-?}   THE BOARD (at-risk arm)"
echo "JOB 4  easyocr:tuned        exit=${JOB4:-?}   Cal's question"
echo "JOB 5  Paddle dev sweep     exit=${JOB5:-?}   expendable footnote"
echo "       NOTE: the sweep writes its report only after ALL configs finish - no checkpointing."
echo "       So exit=124 on Job 5 means ZERO output, not partial. Nothing to salvage."
echo "JOB 6  Arm C reader svtrv2  exit=${JOB6:-?}   expendable bonus"
echo "JOB 7  Arm C reader parseq  exit=${JOB7:-?}   expendable bonus"
echo ""
echo "Jobs 1-3 are Table A. Each ran alone, so each has its own TABLES.md and they get"
echo "assembled into one ranking at 7am. Jobs 5-6 not finishing is expected, not a failure."
echo ""
echo "IN THE MORNING: bash experiments/morning_summary.sh; cat experiments/MORNING_SUMMARY.md"
echo ""
