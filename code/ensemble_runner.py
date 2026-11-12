#!/usr/bin/env python3
"""HPC execution driver for the 50-replicate Douala ensemble.

Loads model definitions and executes precomputation, independent seed tasks,
or aggregate analysis. Standalone analysis blocks in the model source are
not executed by individual Slurm workers.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import traceback
from pathlib import Path


BASELINE_BLOCK_MARKER = "# --- Run baseline scenario"
EXPECTED_CHECKPOINT_VERSION = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-source", required=True)
    p.add_argument("--city_slug", default="douala")
    p.add_argument("--run_id", type=int, default=3000)
    p.add_argument("--mode", choices=["precompute", "worker", "aggregate"], required=True)
    p.add_argument("--model-plots", action="store_true", help="Include the model figure outputs during aggregation.")
    return p.parse_args()


def _exec_model_prefix(source: str, model_path: Path, city_slug: str, run_id: int) -> dict:
    marker_pos = source.find(BASELINE_BLOCK_MARKER)
    if marker_pos < 0:
        raise RuntimeError(f"Required marker not found in {model_path}: {BASELINE_BLOCK_MARKER!r}")
    prefix = source[:marker_pos]
    ns = {
        "__name__": "__douala_model__",
        "__file__": str(model_path),
        "__package__": None,
    }
    old_argv = sys.argv[:]
    try:
        # Model initialisation parses command-line arguments.
        sys.argv = [str(model_path), "--city_slug", city_slug, "--run_id", str(run_id)]
        exec(compile(prefix, str(model_path), "exec"), ns, ns)
    finally:
        sys.argv = old_argv
    if "run_scenario" not in ns:
        raise RuntimeError("Safe prefix load did not define run_scenario().")
    return ns


def _load_later_definitions_and_constants(source: str, model_path: Path, ns: dict) -> None:
    """Load analysis definitions without executing standalone model runs."""
    marker_pos = source.find(BASELINE_BLOCK_MARKER)
    later = source[marker_pos:]
    tree = ast.parse(later, filename=str(model_path))

    # Import model definitions in source order. Ignore optional
    # dependency imports that are irrelevant to the co-occurrence ensemble.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                mod = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(mod)
                exec(compile(mod, str(model_path), "exec"), ns, ns)
            except ImportError as exc:
                print(f"[runner] optional later import skipped: {exc}")

    # Second pass: only literal assignments. This safely recovers labels,
    # numeric plotting constants, cohort dictionaries, and schema versions.
    for node in tree.body:
        target_name = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        if target_name is None or value_node is None:
            continue
        try:
            ns[target_name] = ast.literal_eval(value_node)
        except Exception:
            pass

    # Recover literal dictionary updates used by plotting routines, without
    # execute arbitrary expressions.
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "update":
            continue
        if not isinstance(call.func.value, ast.Name) or len(call.args) != 1:
            continue
        name = call.func.value.id
        if name not in ns or not isinstance(ns.get(name), dict):
            continue
        try:
            payload = ast.literal_eval(call.args[0])
        except Exception:
            continue
        if isinstance(payload, dict):
            ns[name].update(payload)


def _configure_ensemble(ns: dict, args: argparse.Namespace) -> None:
    # The model can derive CITY_SLUG from its full place name during
    # setup with a slugified full PLACE_NAME (e.g.
    # douala_douala_ii_communaute_urbaine_de_douala_wouri_littoral_cameroon).
    # Use the short input slug supplied on
    # the command line.  This changes only output filenames, never model logic,
    # layouts, scenarios, random seeds, or simulation state.
    requested_city_slug = str(args.city_slug).strip()
    ns["CITY_SLUG"] = requested_city_slug
    ns["INPUT_CITY_SLUG"] = requested_city_slug
    print(f"[runner] publication/output CITY_SLUG restored to: {requested_city_slug}")

    required = [
        "_base_intervention_d_dict", "run_cooccurrence_ensemble", "run_scenario",
        "Path", "CITY_OUT", "CITY_SLUG", "RUN_SEED", "RUN_TICKS",
    ]
    missing = [x for x in required if x not in ns]
    if missing:
        raise RuntimeError(f"Safe model load is missing required symbols: {missing}")

    # Preserve the active D parameters from the supplied model. These are copied
    # from the source if literal extraction succeeded; the fallback exactly
    # matches the uploaded model and is used only if the name was not recovered.
    ns.setdefault("ACTIVE_D_PARAMS", {
        "k_geom": 2.40,
        "delta_dest": 0.80,
        "delta_fac": 0.80,
        "lambda_blend": 0.45,
    })

    worker = args.mode == "worker"
    aggregate = args.mode == "aggregate"
    ns["RUN_SOBOL"] = False
    ns["RUN_PARAMETER_SCREENING"] = False
    ns["RUN_MOBILITY_COMPARATOR_MAIN"] = False
    ns["RUN_COOCCURRENCE_ENSEMBLE"] = True
    ns["COOCCURRENCE_N_SEEDS"] = int(os.environ.get("COOCCURRENCE_N_SEEDS", "1" if worker else "50"))
    ns["COOCCURRENCE_SEED_START"] = int(os.environ.get("COOCCURRENCE_SEED_START", "42"))
    ns["COOCCURRENCE_RESUME"] = bool(int(os.environ.get("COOCCURRENCE_RESUME", "1")))
    ns["COOCCURRENCE_WORKER_ONLY"] = worker
    ns["COOCCURRENCE_AGGREGATE_ONLY"] = aggregate
    ns["RUN_PAPER_PLOTS"] = bool(args.model_plots and aggregate)
    ns["RUN_DIAGNOSTIC_PLOTS"] = bool(args.model_plots and aggregate)
    ns["KEEP_LEGACY_PLOTS"] = True
    ns["MIN_COHORT_N_FOR_MAIN_PLOT"] = int(os.environ.get("MIN_COHORT_N_FOR_MAIN_PLOT", "30"))
    ns["COOCCURRENCE_CHECKPOINT_VERSION"] = EXPECTED_CHECKPOINT_VERSION

    # Initialise ensemble constants without executing standalone
    # top-level simulation blocks.
    ns["COOCC_SCENARIOS"] = {
        "baseline": None,
        "intervD_destinations_only": ns["_base_intervention_d_dict"]("destinations_only", "intervD_destinations_only"),
        "intervD_facilities_only": ns["_base_intervention_d_dict"]("sanitation_only", "intervD_facilities_only"),
        "intervD_area_redistribute": ns["_base_intervention_d_dict"]("all", "intervD_area_redistribute"),
    }
    ns["COOCC_ORDER"] = list(ns["COOCC_SCENARIOS"])
    ns.setdefault("COOCC_LABELS", {
        "baseline": "Baseline clustered",
        "intervD_destinations_only": "D1 activity redistribution",
        "intervD_facilities_only": "D2 WASH redistribution",
        "intervD_area_redistribute": "D3 activity + WASH redistribution",
    })
    ns["PAPER_SCENARIO_LABELS"] = dict(ns["COOCC_LABELS"])
    ns.setdefault("FINAL_PAPER_SCENARIO_LABELS", {
        "baseline": "Baseline",
        "intervD_destinations_only": "POI",
        "intervD_facilities_only": "WASH",
        "intervD_area_redistribute": "POI+WASH",
    })

    out = Path(ns["CITY_OUT"]) / "cooccurrence_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    ns["COOCC_OUT"] = out
    ns["PAPER_FIG_DIR"] = out / "paper_figures"
    ns["FINAL_PAPER_MAIN_DIR"] = out / "final_paper_main"
    ns["PAPER_MAIN_DIR"] = ns["PAPER_FIG_DIR"] / "main"
    ns["PAPER_SUPP_DIR"] = ns["PAPER_FIG_DIR"] / "supplement"
    ns["PAPER_DIAG_DIR"] = ns["PAPER_FIG_DIR"] / "diagnostic"
    for d in [ns["PAPER_FIG_DIR"], ns["FINAL_PAPER_MAIN_DIR"], ns["PAPER_MAIN_DIR"], ns["PAPER_SUPP_DIR"], ns["PAPER_DIAG_DIR"]]:
        d.mkdir(parents=True, exist_ok=True)

    # Publication plotting defaults for model figures.
    ns["FINAL_PAPER_DPI"] = 600
    ns.setdefault("FINAL_MAP_CMAP", "turbo")
    ns["SHOW_INTERNAL_MAP_BORDERS"] = False


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_source).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    source = model_path.read_text(encoding="utf-8")

    print(f"[runner] model={model_path}")
    print(f"[runner] mode={args.mode} city={args.city_slug} run_id={args.run_id}")
    print("[runner] loading model definitions for ensemble execution")
    ns = _exec_model_prefix(source, model_path, args.city_slug, args.run_id)
    _load_later_definitions_and_constants(source, model_path, ns)
    if args.mode == "precompute":
        print("[runner] PRECOMPUTE SUCCESS: urban inputs, model definitions, and reusable caches loaded without an epidemic scenario run.")
        return 0
    _configure_ensemble(ns, args)

    print(
        "[runner] ensemble configuration: "
        f"n_seeds={ns['COOCCURRENCE_N_SEEDS']} start={ns['COOCCURRENCE_SEED_START']} "
        f"checkpoint_v={ns['COOCCURRENCE_CHECKPOINT_VERSION']} "
        f"model_plots={ns['RUN_PAPER_PLOTS'] or ns['RUN_DIAGNOSTIC_PLOTS']}"
    )
    ns["run_cooccurrence_ensemble"]()
    print("[runner] SUCCESS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[runner] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
