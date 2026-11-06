"""Long diagnostic summaries retaining replay bounds and every censored cell."""

from __future__ import annotations


def long_measurement(report: dict, prefix: str) -> dict:
    """Record coverage and actual-work censor intervals, including legacy controls."""
    rows = report["runs"]
    return {
        "passed": sum(r[f"{prefix}_status"] == "passed" for r in rows),
        "expected": report["expected_cells"],
        "trainer_seconds_last_recorded_segments": sum(
            r[f"{prefix}_seconds_last_run_segment"] or 0 for r in rows
        ),
        "censored": [
            {
                key: row[key]
                for key in [
                    "case",
                    "seed",
                    f"{prefix}_canonical_censor_epochs",
                    f"{prefix}_censor_epochs_lower",
                    f"{prefix}_censor_epochs_upper",
                    f"{prefix}_final_heldout_accuracy",
                ]
            }
            for row in rows
            if row[f"{prefix}_status"] == "budget_exhausted"
        ],
    }


def long_contrasts(report: dict) -> dict:
    """Count common successes by status, even when their exact cost is uncertain."""
    rows = report["runs"]
    common = [
        r for r in rows if r["baseline_status"] == r["candidate_status"] == "passed"
    ]
    lower = [r["epoch_ratio_lower"] for r in common]
    upper = [r["epoch_ratio_upper"] for r in common]
    return {
        "geometric_mean_epoch_ratio": report["geometric_mean_epoch_ratio"],
        "geometric_mean_epoch_ratio_lower": report["geometric_mean_epoch_ratio_lower"],
        "geometric_mean_epoch_ratio_upper": report["geometric_mean_epoch_ratio_upper"],
        "common_passes": len(common),
        "earlier": report["paired_earlier"],
        "tied": report["paired_tied"],
        "later": report["paired_later"],
        "uncertain_order": len(common)
        - report["paired_earlier"]
        - report["paired_tied"]
        - report["paired_later"],
        "recovered": [
            r
            for r in rows
            if r["baseline_status"] == "budget_exhausted"
            and r["candidate_status"] == "passed"
        ],
        "newly_censored": [
            r
            for r in rows
            if r["baseline_status"] == "passed"
            and r["candidate_status"] == "budget_exhausted"
        ],
        "both_censored": [
            r
            for r in rows
            if r["baseline_status"] == r["candidate_status"] == "budget_exhausted"
        ],
        "worst_observed_ratio_lower": max(lower, default=None),
        "worst_observed_ratio_upper": max(upper, default=None)
        if all(value is not None for value in upper)
        else None,
        "regressed_common": [r for r in common if r["epoch_ratio_lower"] > 1],
    }
