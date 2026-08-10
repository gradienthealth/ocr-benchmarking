"""Phase 6 aggregation — roll per-image score rows up to per-stratum + overall. 🟢 PHI-free.

`score()` (Phase 5) emits raw, unaveraged per-image ingredients; this is where they become
rates and latency percentiles. Two CLAUDE.md invariants are preserved structurally:

- **Hallucination and omission are never averaged together.** The Found axis
  (`found_count`/`omission_count`) and the Added axis (`added_count`) stay in separate keys;
  no combined "accuracy"/"error" number is ever produced.
- **False-redaction rate is the headline** — `sum(false_redaction_count) / sum(keep_total)`,
  KEEP-denominated. CER/WER are rolled up ONLY inside a nested `diagnostic` sub-dict (macro
  mean over the per-image `diagnostic` values); they share no key with the ranking group, so
  a refactor cannot accidentally promote them into the ranking signal — exactly as the scorer
  keeps them quarantined per-image. Phase 7's report reads them from `diagnostic`.

Latency is reported as mean/median/p95, split into the model's own time (`elapsed`, the
primary `run_func` call) and the verifier's time (`verifier_elapsed`) — never merged, so the
gated setup's two costs stay legible. The negative-control hallucination floor is aggregated
over just the confirmed-blank rows.

Grouping key is `row["stratum"]`, which `run_harness` stamps authoritatively from the manifest
(CLAUDE.md rule #7) — so blank controls (zero GT tokens, `None` stratum from `score()`) still
land in their real stratum. Aggregation is an order-independent reduction: shuffling `rows`
yields an identical result.

PHI-free: consumes only the numeric/string ingredients in the score rows — never token text
or pixels.
"""

from __future__ import annotations

import statistics
from typing import Optional


def _latency_stats(values: list[float]) -> Optional[dict]:
    """mean/median/p95 over a list of latencies, or None if empty.

    p95 uses linear interpolation between the two nearest ranks (statistics.quantiles,
    exclusive method); for one value it degenerates to that value.
    """
    if not values:
        return None
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "median": v, "p95": v}
    ordered = sorted(values)
    # 100 cut points -> the 95th is the p95 boundary; interpolates between neighbours.
    p95 = statistics.quantiles(ordered, n=100, method="exclusive")[94]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": float(p95),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    """numerator/denominator, or None when the denominator is 0 (no division by zero)."""
    return (numerator / denominator) if denominator else None


def _group_stats(rows: list[dict]) -> dict:
    """Aggregate one group of score rows (a stratum, or the whole set).

    Every count is an order-independent sum; the Found and Added axes are kept in
    disjoint keys and never combined.
    """
    n = len(rows)
    keep_total = sum(r["keep_total"] for r in rows)
    fr_count = sum(r["false_redaction_count"] for r in rows)
    kem_count = sum(r["keep_exact_match_count"] for r in rows)
    costs = [r["cost"] for r in rows]
    verifier_latencies = [
        r["verifier_elapsed"] for r in rows if r.get("verifier_elapsed") is not None
    ]

    # --- diagnostic-only rollup (CER/WER, char censoring) — NEVER a ranking signal ---
    # Kept in a nested key so it can't be mistaken for a headline metric, mirroring how
    # score() quarantines it per-image. CER/WER are macro-averaged over the images that
    # actually produced a value (jiwer returns None when an image has no matched pairs);
    # char counts are order-independent sums.
    diagnostics = [r.get("diagnostic") or {} for r in rows]
    cers = [d["cer"] for d in diagnostics if d.get("cer") is not None]
    wers = [d["wer"] for d in diagnostics if d.get("wer") is not None]
    keep_chars_total = sum(d.get("keep_chars_total", 0) for d in diagnostics)
    keep_chars_censored = sum(d.get("keep_chars_censored", 0) for d in diagnostics)

    return {
        "n_images": n,
        # --- headline: false redaction (KEEP-denominated) ---
        "false_redaction_count": fr_count,
        "keep_total": keep_total,
        "false_redaction_rate": _rate(fr_count, keep_total),
        # --- KEEP read fidelity ---
        "keep_exact_match_count": kem_count,
        "keep_exact_match_rate": _rate(kem_count, keep_total),
        # --- Found (omission) axis — never blended with Added ---
        "found_count": sum(r["found_count"] for r in rows),
        "omission_count": sum(r["omission_count"] for r in rows),
        "gt_total": sum(r["gt_total"] for r in rows),
        # --- Added (hallucination) axis — never blended with Found ---
        "added_count": sum(r["added_count"] for r in rows),
        # --- latency, split primary vs verifier ---
        "latency": _latency_stats([r["elapsed"] for r in rows]),
        "verifier_latency": _latency_stats(verifier_latencies),
        # --- cost ---
        "cost": {"sum": sum(costs), "mean": (sum(costs) / n) if n else None},
        # --- DIAGNOSTIC ONLY — never a ranking signal (nested, see comment above) ---
        "diagnostic": {
            "cer_mean": statistics.fmean(cers) if cers else None,
            "wer_mean": statistics.fmean(wers) if wers else None,
            "n_cer_images": len(cers),
            "keep_chars_total": keep_chars_total,
            "keep_chars_censored": keep_chars_censored,
            "char_censor_rate": _rate(keep_chars_censored, keep_chars_total),
        },
    }


def aggregate(rows: list[dict]) -> dict:
    """Roll score rows up to per-stratum + overall stats plus the negative-control floor.

    Returns:
        {
          "model_name": <carried from the rows, for Phase 7>,
          "version": <carried from the rows (D-8.3); always non-empty, see Raises>,
          "verifier_model_name": <carried from the rows; None if no row ran a verifier>,
          "verifier_version": <carried from the rows; None if no row ran a verifier>,
          "n_images": int,
          "overall": <group stats over all rows>,
          "per_stratum": {stratum: <group stats>, ...},   # sorted by stratum
          "negative_control": {floor_count, n_control_images, hallucinations_per_image},
        }

    Empty input yields zero counts and None rates throughout — never an exception.

    Raises:
        ValueError: any row carries a blank/missing `version` (D-8.4). Checked before the
            mixed-pair check, which is blind to it: an all-blank batch is one consistent
            pair. `OCROutput` also rejects a blank version at construction, so this is the
            backstop for rows assembled by hand rather than by a runner.
        ValueError: a row ran a verifier (`verifier_elapsed` is not None) but carries a
            blank/missing `verifier_model_name`/`verifier_version` (D-9.1) — same hole as
            above, extended to the second model in a gated run. `run_harness` also rejects
            this at call time; this is the backstop for hand-built rows.
        ValueError: rows come from more than one (model_name, version, verifier_model_name,
            verifier_version) tuple (D-8.3, extended by D-9.1) — an engine or verifier
            version bump invalidates prior results like a gt.csv change, so a mixed batch
            must fail loudly rather than silently average across versions. This also catches
            blending a primary-only run with a gated run: one has verifier identity `None`,
            the other doesn't, so they are never the same tuple.
    """
    if rows:
        # Blank-version check FIRST (D-8.4). The mixed-pair check below cannot catch this:
        # rows that all carry version "" collapse to ONE pair, so a batch with no version
        # information at all looks maximally consistent. Counts/indices only — never row
        # contents — so this message stays PHI-free.
        blank = [i for i, r in enumerate(rows) if not str(r.get("version") or "").strip()]
        if blank:
            raise ValueError(
                f"aggregate(): {len(blank)} of {len(rows)} rows carry no engine version "
                f"(first at row index {blank[0]}). An unversioned row cannot be shown to "
                "come from the same engine build as its neighbours, and an all-blank batch "
                "would pass the (model_name, version) consistency check as one pair "
                "(rule #9)."
            )

        # Same hole, same fix, for the SECOND model in a gated run (D-9.1). Only rows that
        # actually ran a verifier are in scope — `verifier_elapsed` (not the identity
        # fields) is the signal, since a primary-only row legitimately has no verifier
        # identity at all.
        blank_verifier = [
            i
            for i, r in enumerate(rows)
            if r.get("verifier_elapsed") is not None
            and (
                not str(r.get("verifier_model_name") or "").strip()
                or not str(r.get("verifier_version") or "").strip()
            )
        ]
        if blank_verifier:
            raise ValueError(
                f"aggregate(): {len(blank_verifier)} of {len(rows)} rows ran a verifier but "
                f"carry no verifier identity (first at row index {blank_verifier[0]}). A "
                "gated row cannot be shown to come from the same verifier build as its "
                "neighbours without one, for the same reason an unversioned primary row is "
                "rejected above."
            )

        pairs = {
            (
                r.get("model_name"),
                r.get("version"),
                r.get("verifier_model_name"),
                r.get("verifier_version"),
            )
            for r in rows
        }
        if len(pairs) > 1:
            conflicting = sorted(pairs, key=lambda p: tuple(str(x) for x in p))
            raise ValueError(
                "aggregate(): refusing to blend rows from different (model_name, version, "
                f"verifier_model_name, verifier_version) tuples: {conflicting}"
            )

    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)

    control_rows = [r for r in rows if r["negative_control"]]
    floor_count = sum(r["negative_control_floor_count"] for r in control_rows)
    n_control = len(control_rows)

    return {
        "model_name": rows[0].get("model_name") if rows else None,
        "version": rows[0].get("version") if rows else None,
        "verifier_model_name": rows[0].get("verifier_model_name") if rows else None,
        "verifier_version": rows[0].get("verifier_version") if rows else None,
        "n_images": len(rows),
        "overall": _group_stats(rows),
        "per_stratum": {
            stratum: _group_stats(by_stratum[stratum])
            # str() key so a None stratum (shouldn't occur post-stamp) can't break sorting
            for stratum in sorted(by_stratum, key=str)
        },
        "negative_control": {
            "floor_count": floor_count,
            "n_control_images": n_control,
            "hallucinations_per_image": _rate(floor_count, n_control),
        },
    }
