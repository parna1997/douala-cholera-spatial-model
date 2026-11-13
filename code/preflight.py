#!/usr/bin/env python3
"""Verify Douala inputs, environment and required model outputs."""
from __future__ import annotations
import argparse, importlib, json, py_compile, sys
from pathlib import Path

REQUIRED_IMPORTS = [
    "numpy","pandas","scipy","matplotlib","geopandas","shapely","rasterio",
    "networkx","osmnx","joblib","hdbscan","contextily","sklearn","PIL","duckdb","pyarrow",
]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',default=None); p.add_argument('--city_slug',default='douala'); args=p.parse_args()
    root=Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    failures=[]; info={"python":sys.version,"project_root":str(root)}
    for path in [root/'configs'/'cities.csv',root/'inputs'/args.city_slug/'data'/'pop.tif',root/'inputs'/args.city_slug/'data'/'pop.csv',root/'code'/'douala_model.py',root/'code'/'ensemble_runner.py',root/'code'/'postprocess.py',root/'code'/'render_publication_figures.py',root/'code'/'render_figure6_maps.py',root/'code'/'validate_overture_baseline.py']:
        if not path.exists(): failures.append(f'missing required file: {path}')
    cities=root/'configs'/'cities.csv'
    if cities.exists():
        import pandas as pd
        try:
            c=pd.read_csv(cities)
            if 'city_slug' not in c.columns or args.city_slug not in set(c['city_slug'].astype(str)): failures.append(f'{args.city_slug} not found in {cities}')
        except Exception as exc: failures.append(f'cannot read {cities}: {exc}')
    versions={}
    for name in REQUIRED_IMPORTS:
        try:
            mod=importlib.import_module(name); versions[name]=getattr(mod,'__version__','available')
        except Exception as exc: failures.append(f'import failed: {name}: {exc}')
    info['packages']=versions
    for f in ['douala_model.py','ensemble_runner.py','check_outputs.py','postprocess.py','check_figures.py','render_publication_figures.py','render_figure6_maps.py','prepare_population.py','validate_overture_baseline.py']:
        path=root/'code'/f
        if path.exists():
            try: py_compile.compile(str(path),doraise=True)
            except Exception as exc: failures.append(f'Python compile failed: {path}: {exc}')
    model=root/'code'/'douala_model.py'
    if model.exists():
        txt=model.read_text(encoding='utf-8',errors='replace')
        checks={
            'checkpoint_schema_v7':'COOCCURRENCE_CHECKPOINT_VERSION = 7' in txt,
            '48h_endpoint':'cumulative_exposure_48h_pct' in txt,
            'tap_contamination_bag':'"tap_contamination"' in txt,
            'dynamic_contamination_flag':'dynamically_contaminated' in txt,
        }
        info['required_model_markers']=checks
        for k,v in checks.items():
            if not v: failures.append(f'required model marker missing: {k}')
    out=root/'outputs'/args.city_slug/'preflight'; out.mkdir(parents=True,exist_ok=True)
    info['failures']=failures
    (out/'preflight_report.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
    if failures:
        print('\n'.join('FAIL: '+x for x in failures)); raise SystemExit(f'Preflight failed with {len(failures)} problem(s).')
    print('PREFLIGHT PASSED')
    for k,v in versions.items(): print(f'{k}: {v}')
    print(f'Report: {out / "preflight_report.json"}')

if __name__=='__main__': main()
