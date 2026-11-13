#!/usr/bin/env python3
"""Verify publication figure files and pixel dimensions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image

MAIN_REQUIRED = [
    "fig1A.png","fig1B.png","fig1C.png","fig1D.png","fig1E.png","fig1F.png","fig1_legend.png",
    "fig2A.png","fig2B.png","fig2C.png",
    "fig3A.png","fig3B.png","fig3C.png",
    "fig4A.png","fig4B.png","fig4C.png","fig4D.png",
    "fig5A.png","fig5B.png","fig5C.png","fig5D.png","fig5E.png","fig5F.png","fig5G.png","fig5H.png","fig5I.png",
    "fig6A.png","fig6B.png","fig6C.png","fig6D.png","fig6E.png","fig6F.png","fig6_legend.png","fig6_colorbar.png",
]

# Final journal-style exports from the same new simulation ensemble.
# These are written automatically by postprocess.py.
PUBLICATION_REQUIRED = [
    "Figure4a.png", "Figure4b.png", "Figure4c.png",
    "Figure6a.png", "Figure6b.png",
    "Figure6c.png", "Figure6d.png", "Figure6e.png", "Figure6f.png",
    "Figure6_legend.png", "Figure6_colorbar.png",
]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',default=None); p.add_argument('--city_slug',default='douala'); p.add_argument('--run_id',type=int,default=3000); p.add_argument('--min-width-px',type=int,default=3000); args=p.parse_args()
    root=Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    base=root/'outputs'/args.city_slug/f'run_{args.run_id:03d}'/'cooccurrence_ensemble'/'figures'
    main_dir=base/'main'; publication_dir=base/'publication'
    failures=[]; rows=[]
    required=[(main_dir,name) for name in MAIN_REQUIRED]
    required.extend((publication_dir,name) for name in PUBLICATION_REQUIRED)
    for folder,name in required:
        path=folder/name
        if not path.exists(): failures.append(f'missing: {path}'); continue
        try:
            with Image.open(path) as im:
                w,h=im.size; dpi=im.info.get('dpi',(None,None));
            rows.append({'file':str(path.relative_to(base)),'width_px':w,'height_px':h,'dpi_x':dpi[0] if dpi else None,'dpi_y':dpi[1] if dpi else None,'bytes':path.stat().st_size})
            # Legend/colourbar assets are intentionally wide/short and may be under the panel width floor.
            if name not in {'fig1_legend.png','fig6_legend.png','fig6_colorbar.png',
                            'Figure6_legend.png','Figure6_colorbar.png'} and w < args.min_width_px:
                failures.append(f'{name}: width {w}px < {args.min_width_px}px')
        except Exception as exc: failures.append(f'{name}: unreadable PNG ({exc})')
    # PDF versions of all journal panels are part of the public figure output.
    for name in PUBLICATION_REQUIRED:
        pdf=publication_dir/Path(name).with_suffix('.pdf')
        if not pdf.is_file() or pdf.stat().st_size == 0:
            failures.append(f'missing or empty publication PDF: {pdf}')
    import pandas as pd
    base.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(base/'figure_quality_check.csv',index=False)
    summary={'required_figure_assets':len(required),'found':len(rows),'failures':failures,'min_width_px':args.min_width_px}
    (base/'figure_quality_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if failures:
        print('\n'.join('FAIL: '+x for x in failures)); raise SystemExit(f'Figure validation failed with {len(failures)} issue(s).')
    print(f'FIGURE VALIDATION PASSED: {len(rows)}/{len(required)} required main and publication assets present and readable.')
    print(f'Report: {base / "figure_quality_check.csv"}')

if __name__=='__main__': main()
