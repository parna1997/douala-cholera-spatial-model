#!/usr/bin/env python3
"""Render journal Figure 4(a-c) and Figure 6(a-b) from existing ensemble summaries.

Reads aggregate per-seed CSVs without modifying their contents or running
simulations. Figure 6(c-f) is rendered separately from archived spatial
checkpoints by ``render_figure6_maps.py``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCENARIOS = [
    'baseline', 'intervD_destinations_only',
    'intervD_facilities_only', 'intervD_area_redistribute',
]
LABELS = ['Baseline', 'POI', 'WASH', 'POI+WASH']
FEATURES = ['school','market','generic_activity','tap','toilet']
FEATURE_LABELS = ['Schools','Markets','Generic POIs','Water taps','Public toilets']
PALETTE = ['#333333','#A05A00','#005B82','#008267']


def read(name: str, directory: Path) -> pd.DataFrame:
    file = directory / name
    if not file.is_file():
        raise FileNotFoundError(f'Missing required source CSV: {file}')
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError(f'Empty source CSV: {file}')
    return df


def verify_ensemble(outcomes: pd.DataFrame, expected_seeds: int = 50) -> None:
    """Reject incomplete or unexpectedly duplicated scenario/seed tables."""
    needed = {"scenario", "seed"}
    if not needed.issubset(outcomes.columns):
        raise ValueError(f"Outcome CSV lacks {sorted(needed - set(outcomes.columns))}")
    for scenario in SCENARIOS:
        rows = outcomes[outcomes.scenario.eq(scenario)]
        n = rows.seed.nunique()
        if n != expected_seeds or len(rows) != expected_seeds:
            raise ValueError(
                f"{scenario}: expected {expected_seeds} distinct seed rows; "
                f"found {len(rows)} rows and {n} seeds"
            )
    print(f"Verified {expected_seeds} completed seeds for each of four scenarios.")


def export(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('.png','.pdf'):
        fig.savefig(out/(stem+ext), dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def format_label(value: float, sd: float, unit: str = '.1f') -> str:
    return f'{value:{unit}}\n±{sd:{unit}}'


def heatmap(attraction:pd.DataFrame, column:str, label:str, stem:str, out:Path) -> None:
    d=attraction.copy()
    d[column]=pd.to_numeric(d[column],errors='coerce')
    # Report mean and standard deviation across the per-seed summary values.
    g=d.groupby(['facility_type','scenario'])[column].agg(['mean','std'])
    mean=np.full((len(FEATURES),len(SCENARIOS)),np.nan)
    sd=np.full_like(mean,np.nan)
    for i,feat in enumerate(FEATURES):
        for j,scenario in enumerate(SCENARIOS):
            if (feat,scenario) in g.index:
                mean[i,j]=float(g.loc[(feat,scenario),'mean'])
                sd[i,j]=float(g.loc[(feat,scenario),'std'])
    fig,ax=plt.subplots(figsize=(10.4,5.9), layout='constrained')
    finite=mean[np.isfinite(mean)]
    vmin=0.0 if len(finite) else 0.0
    vmax=max(float(finite.max()) if len(finite) else 1.,.0001)
    im=ax.imshow(mean,aspect='auto',cmap='cividis',norm=Normalize(vmin,vmax))
    for i in range(len(FEATURES)):
        for j in range(len(SCENARIOS)):
            val=mean[i,j]
            if not np.isfinite(val):continue
            rgb=im.cmap(im.norm(val))[:3]
            brightness=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            ax.text(j,i,format_label(val,sd[i,j]),ha='center',va='center',fontsize=15,fontweight='semibold',color='black' if brightness>.52 else 'white')
    ax.set_xticks(range(len(LABELS)),LABELS,fontsize=14)
    ax.set_yticks(range(len(FEATURES)),FEATURE_LABELS,fontsize=14)
    ax.tick_params(length=0,pad=8)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2.0)
    ax.tick_params(which='minor',bottom=False,left=False)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.025)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label,fontsize=14,labelpad=10)
    export(fig,out,stem)


def epidemic_curves(ts:pd.DataFrame,out:Path)->None:
    z=ts[ts.metric.eq('cumulative_exposure_pct')].copy()
    fig,ax=plt.subplots(figsize=(9.6,5.9), layout='constrained')
    for scenario,name,color in zip(SCENARIOS,LABELS,PALETTE):
        group=z[z.scenario.eq(scenario)].groupby('tick')['value'].agg(['mean','std']).sort_index()
        if group.empty: raise ValueError(f'No time series for {name}')
        x=group.index.to_numpy(dtype=float)
        avg=group['mean'].to_numpy(dtype=float)
        err=group['std'].fillna(0).to_numpy(dtype=float)
        ax.plot(x,avg,label=name,color=color,linewidth=2.6)
        ax.fill_between(x,avg-err,avg+err,color=color,alpha=.13)
    ax.set_xlabel('Simulation time (h)',fontsize=15)
    ax.set_ylabel('Cumulative S→E events (% of agents)',fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(ncol=2,frameon=False,fontsize=13,loc='upper left')
    ax.grid(alpha=.16)
    ax.set_xlim(0,168)
    export(fig,out,'Figure6a')


def epidemic_peaks(outcomes:pd.DataFrame,out:Path)->None:
    specs=[('peak_6hr_incidence_pct','Peak 6-hour incidence','#00669E'),('peak_active_infectious_pct','Peak active infectious prevalence','#E37B18')]
    fig,ax=plt.subplots(figsize=(9.6,5.9),layout='constrained')
    x=np.arange(4,dtype=float); width=.34
    for k,(col,name,color) in enumerate(specs):
        agg=outcomes.groupby('scenario')[col].agg(['mean','std']).reindex(SCENARIOS)
        ax.bar(x+(k-.5)*width,agg['mean'].to_numpy(),width,yerr=agg['std'].to_numpy(),color=color,capsize=5,error_kw={'elinewidth':1.6,'capthick':1.6},label=name)
    ax.set_xticks(x,LABELS,fontsize=14)
    ax.set_ylabel('Peak burden (% of agents)',fontsize=15)
    ax.tick_params(axis='y',labelsize=13)
    ax.legend(ncol=1,frameon=False,fontsize=13)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    export(fig,out,'Figure6b')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-dir',required=True,type=Path,help='Directory containing existing aggregate per-seed CSV files')
    p.add_argument('--out',required=True,type=Path,help='Destination folder; original CSVs are untouched')
    p.add_argument('--expected-seeds',type=int,default=50,help='Expected completed stochastic seeds per scenario')
    args=p.parse_args()
    files={key:read(f'douala_{key}_per_seed.csv',args.csv_dir) for key in ('attraction','timeseries','outcomes')}
    verify_ensemble(files['outcomes'], args.expected_seeds)
    for col,label,stem in [
        ('visits_per_agent_day','Completed visits per agent per day','Figure4a'),
        ('median_trip_distance_m','Median completed journey distance (m)','Figure4b'),
        ('visits_per_facility','Mean completed visits per used facility','Figure4c'),
    ]:
        heatmap(files['attraction'],col,label,stem,args.out)
    epidemic_curves(files['timeseries'],args.out)
    epidemic_peaks(files['outcomes'],args.out)
    print(f'Created 5 figure panels (PNG+PDF) at {args.out}. These do not include Figure 6(c-f) maps.')

if __name__=='__main__':main()
