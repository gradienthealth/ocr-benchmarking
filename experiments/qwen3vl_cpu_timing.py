#!/usr/bin/env python3
"""Phase 13d — how long does Qwen3-VL take to read ONE crop on this CPU? 🟢 PHI-FREE.

Claude may run this: every crop is drawn in-memory by `tests/synthetic.py` from obviously-fake
tokens ("CMFN-0042", "GRDN1234"). No render, no `gt.csv`, no manifest, no pixel from a study.

    .venv/bin/python experiments/qwen3vl_cpu_timing.py --n 12 --out experiments/qwen3vl_timing.json

(One line on purpose — a backslash-continued paste mangles the flags into escaped spaces.)

WHY THIS EXISTS
---------------------------------------------------------------------------------
No public figure exists for this model's CPU latency, and Arm C's cost is per CROP, not per
image: a frame with thirty tokens is thirty forward passes. At 10 s/crop that is five minutes
for one ultrasound frame, which decides whether the arm runs over the whole scored set or a
stratified subsample. Nobody should plan around a guess, so this measures it before anyone does.

The numbers are a property of THIS box (8 cores, avx2, no avx512_bf16, no AMX) and of the
pinned dtype. They are a floor for planning, not a production latency: a GPU deployment is a
different measurement entirely, exactly as the docTR/PaddleOCR CPU numbers are (D-9.1).

WHAT IT REPORTS BESIDES SECONDS
---------------------------------------------------------------------------------
  verbose_outputs   replies longer than a burned-in token plausibly is. The reader deliberately
                    does NOT strip "The text reads: …" prefaces (see read_qwen3vl.read), so this
                    is the count of reads that would score as misreads for being chatty.
  abstentions       replies that came back as the abstain marker on a crop that HAS text —
                    the false-abstention side of the contract.
  exact_matches     SYNTHETIC accuracy. Printed to show the wiring works, and labelled so it
                    cannot be quoted as this arm's accuracy: Qwen2.5-VL-7B scored 0.962 on
                    synthetic imprinted text and 0.719 on real burned-in text (arXiv 2511.02014).

The JSON carries `reader_config` verbatim, including the pinned model revision, because
`aggregate()` keeps only the config HASH and a hash is not a thing a human can check (rule #9).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.aggregate import _latency_stats  # noqa: E402
from harness.contract import normalize  # noqa: E402
from harness.readers.read_qwen3vl import Qwen3VLReader  # noqa: E402
from harness.reading import crop_for_reading, crop_spec  # noqa: E402
from tests import synthetic  # noqa: E402

VERBOSE_OUTPUT_CHARS = 24
# A burned-in clinical token is short: an accession number, a patient ID, a laterality letter.
# Anything longer than this is the model writing a sentence, not reading a token. A threshold,
# not a truth — it is reported as a count so the raw judgement stays with the reader.


def _crops(n: int):
    """`n` synthetic crops, cycling the fake token pool, through THE pinned crop pipeline.

    Deliberately routed via `crop_for_reading` rather than handing the model a raw image: the
    timing has to be of what the arm actually feeds it (48px tall, padded, RGB), since the
    vision encoder's cost scales with the pixels it receives.
    """
    tokens = synthetic.FAKE_TOKENS
    out = []
    for i in range(n):
        token = tokens[i % len(tokens)]
        image, gt = synthetic.make_synthetic_image((token,), seed=i, image_id=f"synth-{i:03d}")
        out.append((token, crop_for_reading(image, gt[0].bbox)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=12, help="timed crops (default 12)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed crops first (default 1) — the first forward pass pays lazy kernel init",
    )
    parser.add_argument("--threads", type=int, default=None, help="override torch thread count")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    load_at_start = os.getloadavg()[0]
    kwargs = {"torch_num_threads": args.threads} if args.threads else {}
    reader = Qwen3VLReader(**kwargs)

    t0 = perf_counter()
    reader._ensure_loaded()
    load_seconds = perf_counter() - t0

    for _, crop in _crops(args.warmup):
        reader.read(crop)

    seconds: list[float] = []
    verbose = abstentions = exact = 0
    for token, crop in _crops(args.n):
        t0 = perf_counter()
        text = reader.read(crop)
        seconds.append(perf_counter() - t0)
        if not text:
            abstentions += 1
        elif len(text) > VERBOSE_OUTPUT_CHARS:
            verbose += 1
        if normalize(text) == normalize(token):
            exact += 1

    ordered = sorted(seconds)
    # THE harness's latency statistics, not a second implementation. Its p95 interpolates
    # between the two nearest ranks; a hand-rolled nearest-rank p95 returns the max at n=12,
    # and a "p95" here that means something different from the "p95" in every harness result
    # is the kind of quiet mismatch nobody catches until two numbers are compared in a report.
    latency = _latency_stats(seconds)
    report = {
        "arm": "reader",
        "model_name": reader.model_name,
        "version": reader.version,
        "config_id": reader.config_id,
        "config_hash": reader.config_hash(),
        "reader_config": reader.config(),
        "crop_preprocessing": crop_spec(),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            # Recorded because these 8 cores are shared with whatever else is running. A run
            # measured at load ~6 came out ~1.8x slower per crop than the same run on an idle
            # box; without this field the two numbers look like a mystery instead of a queue.
            "load_average_1min_at_start": load_at_start,
            "load_average_1min_at_end": os.getloadavg()[0],
        },
        "n_crops": args.n,
        "warmup_crops": args.warmup,
        "load_seconds": round(load_seconds, 3),
        "seconds_per_crop": {
            **{k: round(v, 3) for k, v in latency.items()},
            "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3),
        },
        "verbose_outputs": verbose,
        "abstentions_on_text_bearing_crops": abstentions,
        "exact_matches_SYNTHETIC_not_an_accuracy_number": exact,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
