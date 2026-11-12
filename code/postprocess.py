#!/usr/bin/env python3
"""Statistical summaries and publication figures for the Douala ensemble.

Input: the 50-seed aggregate CSVs/checkpoints created by douala_model.py.
Output: statistics/, figures/main/, figures/supplement/.

The script performs analysis/visualisation only; it never runs the ABM.
"""
from __future__ import annotations

import argparse
import gzip
import os
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCENARIOS = [
    "baseline",
    "intervD_destinations_only",
    "intervD_facilities_only",
    "intervD_area_redistribute",
]
LABELS = {
    "baseline": "Baseline",
    "intervD_destinations_only": "POI",
    "intervD_facilities_only": "WASH",
    "intervD_area_redistribute": "POI+WASH",
}
FEATURE_LABELS = {
    "generic_activity": "Generic POIs",
    "school": "Schools",
    "market": "Markets",
    "tap": "Water taps",
    "toilet": "Public toilets",
    "activity_supply": "POI supply",
    "wash_supply": "WASH supply",
    "realised_visits": "Realised visits",
}
PAIR_LABELS = {
    "school_near_tap": "School-adjacent taps",
    "market_near_tap": "Market-adjacent taps",
    "home_near_tap": "Home-adjacent taps",
    "activity_near_tap": "Activity-adjacent taps",
    "school_near_toilet": "School-adjacent toilets",
    "market_near_toilet": "Market-adjacent toilets",
    "home_near_toilet": "Home-adjacent toilets",
    "activity_near_toilet": "Activity-adjacent toilets",
    "school_near_market": "Markets near schools",
}
CHECKPOINT_VERSION = 7
DPI = int(os.environ.get("DOUALA_FIGURE_DPI", "450"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None)
    p.add_argument("--city_slug", default="douala")
    p.add_argument("--run_id", type=int, default=3000)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--analysis-seed", type=int, default=20260825)
    return p.parse_args()


def _style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _read_csv(run_dir: Path, city: str, name: str) -> pd.DataFrame:
    path = run_dir / f"{city}_{name}_per_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required aggregate CSV is missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Required aggregate CSV is empty: {path}")
    return df


def _mean_sd(df: pd.DataFrame, group_cols, value_col: str) -> pd.DataFrame:
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    return d.groupby(group_cols, dropna=False)[value_col].agg(["mean", "std", "count"]).reset_index()


def _save(fig, path: Path, vector=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if vector:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)




def _bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.fillna(False).astype(str).str.strip().str.lower().isin({"true","1","yes","y","t"})

def _holms(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted_sorted[rank] = running
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out


def _bootstrap_diff(a, b, n_resamples, rng):
    """CI for mean(a)-mean(b), independent replicate-level resampling."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    vals = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        vals[i] = aa.mean() - bb.mean()
    return tuple(np.quantile(vals, [0.025, 0.975]))


def _permutation_diff(a, b, n_perm, rng):
    """Two-sided unpaired permutation test for difference in replicate means."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = abs(perm[:na].mean() - perm[na:].mean())
        if stat >= obs - 1e-15:
            extreme += 1
    return (extreme + 1.0) / (n_perm + 1.0)


def primary_statistics(outcomes, stats_dir, args):
    field = "cumulative_exposure_48h_pct"
    required = {"scenario", "seed", field, "cumulative_exposure_48h_count", "cumulative_exposure_168h_pct", "cumulative_exposure_168h_count"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcome table lacks required endpoint fields: {sorted(missing)}")
    d = outcomes[list(required)].copy()
    d["scenario_label"] = d["scenario"].map(LABELS)
    d.sort_values(["scenario", "seed"]).to_csv(stats_dir / "primary_endpoint_by_seed.csv", index=False)

    baseline = pd.to_numeric(d.loc[d.scenario.eq("baseline"), field], errors="coerce").dropna().to_numpy()
    if len(baseline) != args.n_seeds:
        raise ValueError(f"Primary endpoint expected {args.n_seeds} Baseline rows; found {len(baseline)}")

    rng_boot = np.random.default_rng(args.analysis_seed)
    rng_perm = np.random.default_rng(args.analysis_seed + 1)
    rows = []
    raw_p = []
    for sc in SCENARIOS[1:]:
        values = pd.to_numeric(d.loc[d.scenario.eq(sc), field], errors="coerce").dropna().to_numpy()
        if len(values) != args.n_seeds:
            raise ValueError(f"Primary endpoint expected {args.n_seeds} rows for {sc}; found {len(values)}")
        diff = float(values.mean() - baseline.mean())
        lo, hi = _bootstrap_diff(values, baseline, args.bootstrap_resamples, rng_boot)
        p = _permutation_diff(values, baseline, args.permutations, rng_perm)
        raw_p.append(p)
        rows.append({
            "comparison": f"{LABELS[sc]} vs Baseline",
            "scenario": sc,
            "baseline_mean_pct": float(baseline.mean()),
            "scenario_mean_pct": float(values.mean()),
            "mean_difference_percentage_points": diff,
            "bootstrap_95ci_low": float(lo),
            "bootstrap_95ci_high": float(hi),
            "permutation_p_raw": float(p),
            "n_baseline_replicates": int(len(baseline)),
            "n_scenario_replicates": int(len(values)),
        })
    adjusted = _holms(raw_p)
    for row, p_adj in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(p_adj)
    comp = pd.DataFrame(rows)
    comp.to_csv(stats_dir / "primary_endpoint_comparisons.csv", index=False)
    (stats_dir / "primary_endpoint_comparisons.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")

    seven = outcomes.groupby("scenario", dropna=False).agg(
        n_replicates=("seed", "nunique"),
        mean_cumulative_168h_count=("cumulative_exposure_168h_count", "mean"),
        sd_cumulative_168h_count=("cumulative_exposure_168h_count", "std"),
        mean_cumulative_168h_pct=("cumulative_exposure_168h_pct", "mean"),
        sd_cumulative_168h_pct=("cumulative_exposure_168h_pct", "std"),
    ).reset_index()
    seven["scenario_label"] = seven["scenario"].map(LABELS)
    seven.to_csv(stats_dir / "seven_day_cumulative_summary.csv", index=False)

    txt = ["Primary endpoint: cumulative logged S→E events at 48 h (% simulated population)",
           f"Bootstrap resamples: {args.bootstrap_resamples}; permutation draws: {args.permutations}; Holm adjustment: 3 Baseline contrasts.",
           "Permutation tests use independent replicate-level label permutations; seed numbers are not assumed to create paired observations.", ""]
    for _, r in comp.iterrows():
        txt.append(f"{r['comparison']}: Δ={r['mean_difference_percentage_points']:.3f} pp, 95% CI {r['bootstrap_95ci_low']:.3f} to {r['bootstrap_95ci_high']:.3f}, Holm p={r['holm_adjusted_p']:.6g}")
    (stats_dir / "primary_endpoint_summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    return comp, seven


def contamination_statistics(contam, stats_dir):
    required = {"scenario","seed","tap_id","contamination_origin","contamination_tick","visits_total","visits_while_contaminated","tap_water_exposures"}
    missing = required - set(contam.columns)
    if missing:
        raise ValueError(f"Tap contamination table missing fields: {sorted(missing)}")
    d = contam.copy()
    for col in ["visits_total", "visits_while_contaminated", "tap_water_exposures", "contamination_tick", "tap_x", "tap_y"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    by_seed = d.groupby(["scenario", "seed", "contamination_origin"], dropna=False).agg(
        n_taps=("tap_id", "size"),
        total_visits=("visits_total", "sum"),
        total_visits_while_contaminated=("visits_while_contaminated", "sum"),
        total_tap_water_exposures=("tap_water_exposures", "sum"),
        mean_contamination_tick=("contamination_tick", "mean"),
    ).reset_index()
    by_seed["exposures_per_1000_contaminated_visits"] = 1000.0 * by_seed["total_tap_water_exposures"] / by_seed["total_visits_while_contaminated"].replace(0, np.nan)
    by_seed.to_csv(stats_dir / "tap_contamination_by_seed_summary.csv", index=False)

    metrics = ["n_taps", "total_visits", "total_visits_while_contaminated", "total_tap_water_exposures", "exposures_per_1000_contaminated_visits", "mean_contamination_tick"]
    agg = by_seed.groupby(["scenario", "contamination_origin"], dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(str(x) for x in c if str(x)) if isinstance(c, tuple) else str(c) for c in agg.columns]
    agg.to_csv(stats_dir / "tap_contamination_scenario_origin_mean_sd.csv", index=False)

    # One row per scenario/seed for the headline counts/fractions.
    total = d.groupby(["scenario","seed"]).size().rename("n_taps").reset_index()
    piv = d.groupby(["scenario","seed","contamination_origin"]).size().unstack(fill_value=0).reset_index()
    for name in ["initial","dynamic","never"]:
        if name not in piv.columns: piv[name] = 0
    headline = total.merge(piv[["scenario","seed","initial","dynamic","never"]], on=["scenario","seed"], how="left")
    headline["pct_initial"] = 100.0 * headline["initial"] / headline["n_taps"]
    headline["pct_dynamic"] = 100.0 * headline["dynamic"] / headline["n_taps"]
    headline["pct_never"] = 100.0 * headline["never"] / headline["n_taps"]
    headline.to_csv(stats_dir / "tap_contamination_headline_by_seed.csv", index=False)
    headline_summary = headline.groupby("scenario").agg(
        n_replicates=("seed","nunique"),
        mean_n_taps=("n_taps","mean"),
        mean_initial_taps=("initial","mean"), sd_initial_taps=("initial","std"),
        mean_dynamic_taps=("dynamic","mean"), sd_dynamic_taps=("dynamic","std"),
        mean_never_taps=("never","mean"), sd_never_taps=("never","std"),
        mean_pct_dynamic=("pct_dynamic","mean"), sd_pct_dynamic=("pct_dynamic","std"),
    ).reset_index()
    headline_summary["scenario_label"] = headline_summary["scenario"].map(LABELS)
    headline_summary.to_csv(stats_dir / "tap_contamination_headline_mean_sd.csv", index=False)

    # Frequency with which the same fixed tap dynamically contaminates across replicates.
    freq = d.groupby(["scenario","tap_id"], dropna=False).agg(
        n_replicates=("seed","nunique"),
        n_dynamic=("dynamically_contaminated", lambda x: int(_bool_series(pd.Series(x)).sum())),
        n_initial=("initially_contaminated", lambda x: int(_bool_series(pd.Series(x)).sum())),
        tap_x=("tap_x","mean"), tap_y=("tap_y","mean"),
    ).reset_index()
    freq["dynamic_frequency_pct"] = 100.0 * freq["n_dynamic"] / freq["n_replicates"].replace(0, np.nan)
    freq["initial_frequency_pct"] = 100.0 * freq["n_initial"] / freq["n_replicates"].replace(0, np.nan)
    freq.to_csv(stats_dir / "dynamic_contamination_frequency_by_tap.csv", index=False)
    return by_seed, headline_summary, freq


def _dot_interval(df, category_col, value_col, out_path, xlabel, category_order=None, legend=True, figsize=(10.5,7.5)):
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    g = d.groupby([category_col, "scenario"], dropna=False)[value_col].agg(["mean","std","count"]).reset_index()
    cats = category_order or [x for x in g[category_col].dropna().astype(str).unique()]
    ybase = np.arange(len(cats), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(SCENARIOS))
    fig, ax = plt.subplots(figsize=figsize)
    for off, sc in zip(offsets, SCENARIOS):
        z = g[g.scenario.eq(sc)].set_index(category_col)
        means=[]; sds=[]
        for cat in cats:
            if cat in z.index:
                rr=z.loc[cat]
                if isinstance(rr, pd.DataFrame): rr=rr.iloc[0]
                means.append(rr["mean"]); sds.append(rr["std"] if np.isfinite(rr["std"]) else 0.0)
            else:
                means.append(np.nan); sds.append(np.nan)
        ax.errorbar(means, ybase+off, xerr=sds, fmt="o", capsize=3, linewidth=1.2, markersize=6, label=LABELS[sc])
    ax.set_yticks(ybase)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.22)
    if legend:
        ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_path)


def _scenario_feature_points(df, value_col, out_path, ylabel, feature_order, figsize=(10.5,7.0)):
    d=df.copy(); d[value_col]=pd.to_numeric(d[value_col], errors="coerce")
    g=d.groupby(["scenario","facility_type"])[value_col].agg(["mean","std"]).reset_index()
    x=np.arange(len(SCENARIOS),dtype=float); offs=np.linspace(-0.24,0.24,len(feature_order))
    fig,ax=plt.subplots(figsize=figsize)
    for off,ft in zip(offs,feature_order):
        z=g[g.facility_type.eq(ft)].set_index("scenario").reindex(SCENARIOS)
        ax.errorbar(x+off,z["mean"],yerr=z["std"].fillna(0),fmt="o",capsize=3,label=FEATURE_LABELS.get(ft,ft),markersize=6)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCENARIOS])
    ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.22); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); _save(fig,out_path)


def _load_layout(ckdir: Path, scenario: str):
    p=ckdir/f"v{CHECKPOINT_VERSION}_layout_{scenario}.pkl.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    with gzip.open(p,"rb") as f: return pickle.load(f)


def _active_layout_layer(layout, name):
    mapping={
        "generic_activity":"generic_active", "school":"schools_active", "market":"markets_active",
        "tap":"taps_active", "toilet":"toilets_active", "hubs":"inner_hubs_active", "outer":"outer_cells_active",
    }
    return layout.get(mapping.get(name,name), None) if isinstance(layout,dict) else None


def plot_layout_map(layout, scenario, path):
    fig,ax=plt.subplots(figsize=(9.2,9.2))
    outer=_active_layout_layer(layout,"outer"); hubs=_active_layout_layer(layout,"hubs")
    if outer is not None and not getattr(outer,"empty",True): outer.boundary.plot(ax=ax,linewidth=.7,alpha=.55)
    if hubs is not None and not getattr(hubs,"empty",True): hubs.boundary.plot(ax=ax,linewidth=1.0,alpha=.75)
    init=layout.get("initial_agents",pd.DataFrame()) if isinstance(layout,dict) else pd.DataFrame()
    if init is not None and not getattr(init,"empty",True) and {"home_x","home_y"}.issubset(init.columns):
        ax.scatter(init.home_x,init.home_y,s=4,alpha=.12,linewidths=0)
    gen=_active_layout_layer(layout,"generic_activity")
    if gen is not None and not getattr(gen,"empty",True):
        try: ax.hexbin(gen.geometry.x,gen.geometry.y,gridsize=40,mincnt=1,alpha=.22,linewidths=0)
        except Exception: ax.scatter(gen.geometry.x,gen.geometry.y,s=4,alpha=.12)
    specs=[("school","^",46),("market","s",42),("tap","o",48),("toilet","x",52)]
    for ft,mk,size in specs:
        g=_active_layout_layer(layout,ft)
        if g is not None and not getattr(g,"empty",True): ax.scatter(g.geometry.x,g.geometry.y,marker=mk,s=size,alpha=.85,linewidths=.7)
    ext=outer if outer is not None and not getattr(outer,"empty",True) else hubs
    if ext is not None and not getattr(ext,"empty",True):
        minx,miny,maxx,maxy=ext.total_bounds; pad=.03*max(maxx-minx,maxy-miny); ax.set_xlim(minx-pad,maxx+pad); ax.set_ylim(miny-pad,maxy+pad)
    ax.set_aspect("equal",adjustable="box"); ax.axis("off"); fig.tight_layout(); _save(fig,path,vector=False)


def make_main_figures(run_dir, main_dir, supp_dir, tables, ckdir):
    # Figure 1 maps and legend
    map_files={"baseline":"fig1A.png","intervD_destinations_only":"fig1B.png","intervD_facilities_only":"fig1C.png","intervD_area_redistribute":"fig1D.png"}
    layouts={}
    for sc in SCENARIOS:
        layouts[sc]=_load_layout(ckdir,sc); plot_layout_map(layouts[sc],sc,main_dir/map_files[sc])
    fig,ax=plt.subplots(figsize=(10,1.25)); ax.axis("off")
    handles=[Patch(alpha=.22,label="Generic POI density"),Line2D([0],[0],marker="^",linestyle="None",label="Schools",markersize=9),Line2D([0],[0],marker="s",linestyle="None",label="Markets",markersize=9),Line2D([0],[0],marker="o",linestyle="None",label="Water taps",markersize=9),Line2D([0],[0],marker="x",linestyle="None",label="Public toilets",markersize=9)]
    ax.legend(handles=handles,loc="center",ncol=5,frameon=False); _save(fig,main_dir/"fig1_legend.png",vector=True)

    uf=tables["urban_form"].copy(); uf=uf[uf.layer.isin(["activity_supply","wash_supply","realised_visits"])].copy(); uf["category"]=uf.layer.map(FEATURE_LABELS)
    _dot_interval(uf,"category","top1_share_pct",main_dir/"fig1E.png","Largest-hub share (%)",["POI supply","WASH supply","Realised visits"],figsize=(10.5,5.8))
    phys=tables["physical"].copy(); phys=phys[np.isclose(pd.to_numeric(phys.radius_m,errors="coerce"),500)].copy(); phys=phys[phys.feature_A.isin(["school","market"]) & phys.feature_B.isin(["tap","toilet"])].copy(); phys["category"]=phys.apply(lambda r:f"{FEATURE_LABELS.get(r.feature_A,r.feature_A)} → {FEATURE_LABELS.get(r.feature_B,r.feature_B)}",axis=1)
    _dot_interval(phys,"category","pct_A_with_B_in_radius",main_dir/"fig1F.png","First-listed POIs with second-listed WASH within 500 m (%)",figsize=(10.5,6.2))

    # Figure 2
    attraction=tables["attraction"]
    fts=["generic_activity","school","market","tap","toilet"]
    _scenario_feature_points(attraction,"visits_per_agent_day",main_dir/"fig2A.png","Completed visits per agent per day",fts)
    _scenario_feature_points(attraction,"median_trip_distance_m",main_dir/"fig2B.png","Median completed journey distance (m)",fts)
    _scenario_feature_points(attraction,"visits_per_facility",main_dir/"fig2C.png","Mean visits per used facility",fts)

    # Figure 3: local context burden
    pb=tables["pair_burden"].copy(); pb["category"]=pb.pair.map(PAIR_LABELS).fillna(pb.pair)
    pair_order=[PAIR_LABELS[x] for x in ["school_near_tap","market_near_tap","home_near_tap","activity_near_tap"]]
    taps=pb[pb.target_facility_type.eq("tap")].copy()
    _dot_interval(taps,"category","mean_visits_per_facility",main_dir/"fig3A.png","Mean completed visits per context-selected tap",pair_order,figsize=(10.5,6.4))
    _dot_interval(taps,"category","mean_tap_water_exposures_per_facility",main_dir/"fig3B.png","Mean direct tap-water S→E events per context-selected tap",pair_order,figsize=(10.5,6.4))
    _dot_interval(taps,"category","mean_exposures_per_1000_visits",main_dir/"fig3C.png","Direct tap-water S→E events per 1,000 visits",pair_order,figsize=(10.5,6.4))

    # Figure 4: typology converted from dense heatmap to large dot-and-interval panels.
    typ=tables["tap_typology"].copy(); order=["High residential / high activity","High residential / low activity","Low residential / high activity","Low residential / low activity"]
    _dot_interval(typ,"tap_typology","n_taps",main_dir/"fig4A.png","Number of taps",order,figsize=(10.5,6.4))
    _dot_interval(typ,"tap_typology","mean_visits_per_tap",main_dir/"fig4B.png","Mean completed visits per tap",order,figsize=(10.5,6.4))
    _dot_interval(typ,"tap_typology","mean_tap_water_exposures_per_tap",main_dir/"fig4C.png","Mean direct tap-water S→E events per tap",order,figsize=(10.5,6.4))
    _dot_interval(typ,"tap_typology","mean_exposures_per_1000_visits",main_dir/"fig4D.png","Direct tap-water S→E events per 1,000 visits",order,figsize=(10.5,6.4))

    # Figure 5A-C: spatial POI-WASH proximity cohorts.
    prox=tables["activity_facility_cohort"].copy()
    prox["category"]=prox.apply(lambda r:f"{FEATURE_LABELS.get(r.facility_type,r.facility_type)}: {r.proximity_class}",axis=1)
    prox_order=[]
    for ft in ["school","market"]:
        for ptype in ["tap","toilet"]:
            for near in [f"{ptype} within 500 m",f"no {ptype} within 500 m"]:
                prox_order.append(f"{FEATURE_LABELS.get(ft,ft)}: {near}")
    _dot_interval(prox,"category","exposed_by_day2_pct",main_dir/"fig5A.png","Initially susceptible cohort exposed by day 2 (%)",prox_order,figsize=(11.5,9.8))
    _dot_interval(prox,"category","exposed_by_day3_pct",main_dir/"fig5B.png","Initially susceptible cohort exposed by day 3 (%)",prox_order,figsize=(11.5,9.8))
    _dot_interval(prox,"category","median_time_to_exposure_days",main_dir/"fig5C.png","Median time to early exposure (days)",prox_order,figsize=(11.5,9.8))

    # Figure 5D-F: experienced POI->tap within 6h cohort. Use horizon-specific rows.
    risk=tables["risk"].copy(); expseq=risk[risk.risk_cohort.eq("Activity → tap within 6 h")].copy(); expseq["category"]="POI visit followed by tap within 6 h"
    d2=expseq[expseq.horizon_day.eq(2)]; d3=expseq[expseq.horizon_day.eq(3)]
    _dot_interval(d2,"category","risk_day2_pct",main_dir/"fig5D.png","Initially susceptible cohort exposed by day 2 (%)",["POI visit followed by tap within 6 h"],figsize=(10.5,4.2))
    _dot_interval(d3,"category","risk_day3_pct",main_dir/"fig5E.png","Initially susceptible cohort exposed by day 3 (%)",["POI visit followed by tap within 6 h"],figsize=(10.5,4.2))
    _dot_interval(d3,"category","median_time_to_exposure_days",main_dir/"fig5F.png","Median time to early exposure (days)",["POI visit followed by tap within 6 h"],figsize=(10.5,4.2))

    # Figure 5G-I: role-specific same-day pairings and their explicit comparators.
    rr=tables["role_risk"].copy(); rr=rr[rr.analysis_type.eq("role_specific")].copy(); rr["category"]=rr.risk_cohort
    role_order=["School + tap on same day","School without same-day tap","Market + tap on same day","Market without same-day tap","Activity + tap on same day","Activity without same-day tap"]
    _dot_interval(rr,"category","risk_day2_pct",main_dir/"fig5G.png","Initially susceptible cohort exposed by day 2 (%)",role_order,figsize=(11.5,8.0))
    _dot_interval(rr,"category","risk_day3_pct",main_dir/"fig5H.png","Initially susceptible cohort exposed by day 3 (%)",role_order,figsize=(11.5,8.0))
    _dot_interval(rr,"category","median_time_to_exposure_days",main_dir/"fig5I.png","Median time to exposure (days)",role_order,figsize=(11.5,8.0))

    # Figure 6A cumulative exposure mean ± SD trajectories.
    ts=tables["timeseries"].copy(); z=ts[ts.metric.eq("cumulative_exposure_pct")].copy(); fig,ax=plt.subplots(figsize=(10.8,7.0))
    for sc in SCENARIOS:
        g=z[z.scenario.eq(sc)].groupby("tick")["value"].agg(["mean","std"])
        x=(g.index.to_numpy(dtype=float)+1.0)/24.0
        ax.plot(x,g["mean"],linewidth=2.0,label=LABELS[sc]); ax.fill_between(x,(g["mean"]-g["std"]).to_numpy(),(g["mean"]+g["std"]).to_numpy(),alpha=.16)
    ax.set_xlabel("Simulation day"); ax.set_ylabel("Cumulative logged S→E events (% of simulated population)"); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); _save(fig,main_dir/"fig6A.png")

    # Figure 6B two peak outcomes.
    out=tables["outcomes"].copy(); metrics=[("peak_6hr_incidence_pct","Peak 6-hour incidence"),("peak_active_infectious_pct","Peak active infectious prevalence")]
    fig,ax=plt.subplots(figsize=(10.5,7.0)); x=np.arange(len(SCENARIOS),dtype=float); offs=[-.12,.12]
    for off,(m,lab) in zip(offs,metrics):
        g=out.groupby("scenario")[m].agg(["mean","std"]).reindex(SCENARIOS)
        ax.errorbar(x+off,g["mean"],yerr=g["std"],fmt="o",capsize=4,markersize=7,label=lab)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCENARIOS]); ax.set_ylabel("Percentage of simulated population"); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False); fig.tight_layout(); _save(fig,main_dir/"fig6B.png")

    # Figure 6 C-F mean E+I density at each scenario-specific peak.
    peak_means={}; vmax=0.0
    for sc in SCENARIOS:
        arrs=[]
        for seed in sorted(tables["outcomes"].seed.unique()):
            cp=ckdir/f"v{CHECKPOINT_VERSION}_seed_{int(seed):03d}_{sc}.pkl.gz"
            with gzip.open(cp,"rb") as f: payload=pickle.load(f)
            arr=np.asarray(payload["spatial"]["peak_ei"],dtype=float); arrs.append(arr)
        mean=np.mean(np.stack(arrs),axis=0); peak_means[sc]=mean; vmax=max(vmax,float(np.nanmax(mean)))
    map_names={"baseline":"fig6C.png","intervD_destinations_only":"fig6D.png","intervD_facilities_only":"fig6E.png","intervD_area_redistribute":"fig6F.png"}
    for sc in SCENARIOS:
        layout=layouts[sc]; ext=_active_layout_layer(layout,"outer"); ext=ext if ext is not None and not getattr(ext,"empty",True) else _active_layout_layer(layout,"hubs")
        extent=list(ext.total_bounds) if ext is not None and not getattr(ext,"empty",True) else None
        if extent is not None: extent=[extent[0],extent[2],extent[1],extent[3]]
        fig,ax=plt.subplots(figsize=(9.2,9.2)); ax.imshow(peak_means[sc],origin="lower",extent=extent,cmap="turbo",vmin=0,vmax=max(vmax,1e-12))
        for ft,mk,size in [("school","^",32),("market","s",30),("tap","o",34),("toilet","x",38)]:
            g=_active_layout_layer(layout,ft)
            if g is not None and not getattr(g,"empty",True): ax.scatter(g.geometry.x,g.geometry.y,marker=mk,s=size,alpha=.75,linewidths=.6)
        if ext is not None and not getattr(ext,"empty",True): ext.boundary.plot(ax=ax,linewidth=.6,alpha=.5)
        ax.set_aspect("equal",adjustable="box"); ax.axis("off"); fig.tight_layout(); _save(fig,main_dir/map_names[sc],vector=False)
    fig,ax=plt.subplots(figsize=(9.5,1.2)); ax.axis("off"); handles=[Line2D([0],[0],marker="^",linestyle="None",label="Schools",markersize=9),Line2D([0],[0],marker="s",linestyle="None",label="Markets",markersize=9),Line2D([0],[0],marker="o",linestyle="None",label="Water taps",markersize=9),Line2D([0],[0],marker="x",linestyle="None",label="Public toilets",markersize=9)]; ax.legend(handles=handles,loc="center",ncol=4,frameon=False); _save(fig,main_dir/"fig6_legend.png")
    fig,ax=plt.subplots(figsize=(7.5,1.3)); ax.set_visible(False); sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,max(vmax,1e-12)),cmap="turbo"); sm.set_array([]); cbar=fig.colorbar(sm,ax=ax,orientation="horizontal",fraction=.85,pad=.15); cbar.set_label("Mean E+I density at scenario-specific peak"); _save(fig,main_dir/"fig6_colorbar.png")


def make_contamination_figures(by_seed, freq, supp_dir):
    # Headline counts by contamination origin.
    d=by_seed.copy(); d["category"]=d["contamination_origin"].map({"initial":"Initially contaminated","dynamic":"Dynamically contaminated","never":"Never contaminated"})
    _dot_interval(d,"category","n_taps",supp_dir/"supp_tap_contamination_counts.png","Number of public taps",["Initially contaminated","Dynamically contaminated","Never contaminated"],figsize=(10.5,5.8))
    _dot_interval(d,"category","total_visits_while_contaminated",supp_dir/"supp_tap_contamination_use.png","Completed visits while tap was contaminated",["Initially contaminated","Dynamically contaminated"],figsize=(10.5,5.5))
    _dot_interval(d,"category","total_tap_water_exposures",supp_dir/"supp_tap_contamination_exposure.png","Direct tap-water S→E events",["Initially contaminated","Dynamically contaminated"],figsize=(10.5,5.5))
    # One large dynamic-frequency map per scenario.
    for sc in SCENARIOS:
        z=freq[freq.scenario.eq(sc)].dropna(subset=["tap_x","tap_y"]).copy()
        fig,ax=plt.subplots(figsize=(9.2,8.2)); pts=ax.scatter(z.tap_x,z.tap_y,c=z.dynamic_frequency_pct,s=65,cmap="viridis",vmin=0,vmax=100,edgecolors="none"); cbar=fig.colorbar(pts,ax=ax); cbar.set_label("Replicates dynamically contaminated (%)"); ax.set_xlabel("Projected x (m)"); ax.set_ylabel("Projected y (m)"); ax.set_aspect("equal",adjustable="box"); fig.tight_layout(); _save(fig,supp_dir/f"supp_dynamic_contamination_frequency_{LABELS[sc].lower().replace('+','_plus_')}.png")


def write_figure_manifest(main_dir, supp_dir):
    rows=[]
    for section,folder in [("main",main_dir),("supplement",supp_dir)]:
        for p in sorted(folder.glob("*.png")):
            rows.append({"section":section,"filename":p.name,"bytes":p.stat().st_size,"pdf_exists":p.with_suffix('.pdf').exists()})
    pd.DataFrame(rows).to_csv(main_dir.parent/"figure_manifest.csv",index=False)


def main():
    args=parse_args(); _style()
    root=Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    run_dir=root/"outputs"/args.city_slug/f"run_{args.run_id:03d}"/"cooccurrence_ensemble"
    ckdir=run_dir/"checkpoints"
    if not run_dir.exists(): raise SystemExit(f"Run directory missing: {run_dir}")
    stats_dir=run_dir/"statistics"; main_dir=run_dir/"figures"/"main"; supp_dir=run_dir/"figures"/"supplement"
    for d in [stats_dir,main_dir,supp_dir]: d.mkdir(parents=True,exist_ok=True)

    names=["outcomes","timeseries","urban_form","physical","attraction","pair_burden","tap_typology","activity_facility_cohort","risk","role_risk","tap_contamination"]
    tables={name:_read_csv(run_dir,args.city_slug,name) for name in names}
    # Strict 50-seed check before any inference/figures.
    for sc in SCENARIOS:
        n=tables["outcomes"].loc[tables["outcomes"].scenario.eq(sc),"seed"].nunique()
        if n != args.n_seeds: raise SystemExit(f"Expected {args.n_seeds} outcome replicates for {sc}, found {n}")

    comp,seven=primary_statistics(tables["outcomes"],stats_dir,args)
    by_seed,headline,freq=contamination_statistics(tables["tap_contamination"],stats_dir)
    make_main_figures(run_dir,main_dir,supp_dir,tables,ckdir)
    make_contamination_figures(by_seed,freq,supp_dir)
    write_figure_manifest(main_dir,supp_dir)

    metadata={
        "city_slug":args.city_slug,"run_id":args.run_id,"seed_start":args.seed_start,"n_seeds":args.n_seeds,
        "primary_endpoint":"cumulative logged S→E events at 48 h (% simulated population)",
        "bootstrap_resamples":args.bootstrap_resamples,"permutations":args.permutations,"holm_comparisons":3,
        "permutation_design":"unpaired replicate-level label permutation",
        "figure_dpi":DPI,
    }
    (stats_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"[postprocess] statistics → {stats_dir}")
    print(f"[postprocess] main figures → {main_dir}")
    print(f"[postprocess] supplementary figures → {supp_dir}")
    print("[postprocess] SUCCESS")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"[postprocess] FAILED: {exc}",file=sys.stderr)
        raise
