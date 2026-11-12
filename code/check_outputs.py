#!/usr/bin/env python3
"""Validate the Douala ensemble checkpoints and aggregate outputs."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = [
    "baseline",
    "intervD_destinations_only",
    "intervD_facilities_only",
    "intervD_area_redistribute",
]
REQUIRED_BAGS = {
    "physical", "urban_form", "attraction", "risk", "role_risk", "agent_risk",
    "tap_level", "tap_contamination", "pair_burden", "tap_typology",
    "activity_facility_cohort", "school_wash_alignment", "school_exposure_pathway",
    "outcomes", "timeseries", "paper_summary",
}
REQUIRED_OUTCOME_FIELDS = {
    "scenario", "seed", "n_agents",
    "cumulative_exposure_48h_count", "cumulative_exposure_48h_pct",
    "cumulative_exposure_168h_count", "cumulative_exposure_168h_pct",
    "peak_6hr_incidence_pct", "peak_active_infectious_pct",
}
REQUIRED_CONTAM_FIELDS = {
    "scenario", "seed", "tap_id", "initially_contaminated",
    "dynamically_contaminated", "contamination_origin", "contamination_tick",
    "final_contaminated", "visits_total", "visits_while_contaminated",
    "tap_water_exposures", "tap_water_exposures_per_1000_contaminated_visits",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    p.add_argument("--city_slug", default="douala")
    p.add_argument("--run_id", type=int, default=3000)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--checkpoint-version", type=int, default=7)
    p.add_argument("--require-aggregate-csvs", action="store_true")
    p.add_argument("--write-report", action="store_true")
    return p.parse_args()


def load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    run_dir = root / "outputs" / args.city_slug / f"run_{args.run_id:03d}" / "cooccurrence_ensemble"
    ckdir = run_dir / "checkpoints"
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    failures = []
    rows = []
    if not ckdir.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {ckdir}")

    for seed in seeds:
        for scenario in SCENARIOS:
            path = ckdir / f"v{args.checkpoint_version}_seed_{seed:03d}_{scenario}.pkl.gz"
            status = "ok"
            detail = "complete"
            n_outcome = 0
            n_contam = 0
            if not path.exists():
                status, detail = "failed", "missing checkpoint"
            else:
                try:
                    payload = load_pickle(path)
                    if not isinstance(payload, dict):
                        raise TypeError("checkpoint payload is not a dict")
                    bags = payload.get("bags", {})
                    missing_bags = sorted(REQUIRED_BAGS - set(bags))
                    if missing_bags:
                        raise ValueError(f"missing bags: {missing_bags}")
                    spatial = payload.get("spatial", {})
                    for key in ["exposure", "initial_ei", "peak_ei", "final_ei", "od"]:
                        if key not in spatial:
                            raise ValueError(f"missing spatial field: {key}")

                    outcomes = pd.DataFrame(bags.get("outcomes", []))
                    n_outcome = len(outcomes)
                    if len(outcomes) != 1:
                        raise ValueError(f"expected 1 outcome row, found {len(outcomes)}")
                    missing = REQUIRED_OUTCOME_FIELDS - set(outcomes.columns)
                    if missing:
                        raise ValueError(f"outcome fields missing: {sorted(missing)}")
                    row = outcomes.iloc[0]
                    if str(row["scenario"]) != scenario or int(row["seed"]) != seed:
                        raise ValueError("outcome scenario/seed mismatch")
                    for field in ["cumulative_exposure_48h_count", "cumulative_exposure_48h_pct", "cumulative_exposure_168h_count", "cumulative_exposure_168h_pct"]:
                        value = pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]
                        if not np.isfinite(value):
                            raise ValueError(f"non-finite primary outcome field: {field}")
                    p48 = float(row["cumulative_exposure_48h_pct"])
                    p168 = float(row["cumulative_exposure_168h_pct"])
                    if p48 < -1e-9 or p168 < -1e-9 or p48 > p168 + 1e-9:
                        raise ValueError(f"endpoint ordering invalid: 48h={p48}, 168h={p168}")

                    contam = pd.DataFrame(bags.get("tap_contamination", []))
                    n_contam = len(contam)
                    if contam.empty:
                        raise ValueError("tap_contamination bag is empty")
                    missing = REQUIRED_CONTAM_FIELDS - set(contam.columns)
                    if missing:
                        raise ValueError(f"contamination fields missing: {sorted(missing)}")
                    if set(contam["contamination_origin"].dropna().astype(str)) - {"initial", "dynamic", "never"}:
                        raise ValueError("unexpected contamination_origin value")
                    # A dynamic tap must end contaminated; an initial tap must be final contaminated.
                    if ((contam["dynamically_contaminated"].astype(bool)) & (~contam["final_contaminated"].astype(bool))).any():
                        raise ValueError("dynamic contamination row ends uncontaminated")
                    if ((contam["initially_contaminated"].astype(bool)) & (~contam["final_contaminated"].astype(bool))).any():
                        raise ValueError("initially contaminated row ends uncontaminated")
                except Exception as exc:
                    status, detail = "failed", str(exc)
            if status != "ok":
                failures.append(f"seed={seed} scenario={scenario}: {detail}")
            rows.append({
                "seed": seed, "scenario": scenario, "status": status,
                "detail": detail, "checkpoint": str(path),
                "n_outcome_rows": n_outcome, "n_tap_contamination_rows": n_contam,
            })

    report = pd.DataFrame(rows)
    print(report.groupby("status").size().to_string())
    print(f"Expected checkpoints: {args.n_seeds * len(SCENARIOS)}")
    print(f"Found valid checkpoints: {(report['status'] == 'ok').sum()}")

    if args.require_aggregate_csvs:
        required_csvs = [
            run_dir / f"{args.city_slug}_outcomes_per_seed.csv",
            run_dir / f"{args.city_slug}_tap_contamination_per_seed.csv",
            run_dir / f"{args.city_slug}_physical_per_seed.csv",
            run_dir / f"{args.city_slug}_urban_form_per_seed.csv",
            run_dir / f"{args.city_slug}_attraction_per_seed.csv",
            run_dir / f"{args.city_slug}_pair_burden_per_seed.csv",
            run_dir / f"{args.city_slug}_tap_typology_per_seed.csv",
            run_dir / f"{args.city_slug}_activity_facility_cohort_per_seed.csv",
            run_dir / f"{args.city_slug}_risk_per_seed.csv",
            run_dir / f"{args.city_slug}_role_risk_per_seed.csv",
            run_dir / f"{args.city_slug}_timeseries_per_seed.csv",
        ]
        for path in required_csvs:
            if not path.exists() or path.stat().st_size == 0:
                # Give a diagnostic if the model's slugified full
                # place name was accidentally used as the CSV prefix. This is
                # an output-naming problem, not a failed simulation ensemble.
                suffix = path.name.split(f"{args.city_slug}_", 1)[-1]
                alternatives = sorted(run_dir.glob(f"*_{suffix}"))
                if alternatives:
                    alt_names = ", ".join(p.name for p in alternatives[:5])
                    failures.append(
                        f"missing expected aggregate CSV: {path}; found same table under alternate city prefix: {alt_names}"
                    )
                else:
                    failures.append(f"missing/empty aggregate CSV: {path}")

    if args.write_report:
        out = run_dir / "validation"
        out.mkdir(parents=True, exist_ok=True)
        report.to_csv(out / "checkpoint_validation.csv", index=False)
        summary = {
            "city_slug": args.city_slug,
            "run_id": args.run_id,
            "checkpoint_version": args.checkpoint_version,
            "seed_start": args.seed_start,
            "n_seeds": args.n_seeds,
            "expected_checkpoints": args.n_seeds * len(SCENARIOS),
            "valid_checkpoints": int((report["status"] == "ok").sum()),
            "failures": failures,
        }
        (out / "checkpoint_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote validation report to {out}")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for item in failures[:50]:
            print(" - " + item, file=sys.stderr)
        if len(failures) > 50:
            print(f" - ... plus {len(failures)-50} more", file=sys.stderr)
        raise SystemExit(f"Ensemble validation failed with {len(failures)} problem(s).")

    print("VALIDATION PASSED: all requested ensemble checkpoints are complete and schema-valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
