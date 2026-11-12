#!/usr/bin/env python3
"""External spatial plausibility check for the synthetic Douala Baseline.

The script compares the fixed synthetic Baseline layout with mapped features from
a runtime-selected Overture Maps release using the *same model hub polygons*.
Default: discover the latest available release via the official Overture STAC.
Reference dataset: Overture release 2026-08-19.0, schema v1.18.0.
Mapped feature counts and spatial measures depend on the selected release.
It is deliberately a plausibility check, not a claim that Overture is complete
ground truth or that model facilities should match mapped sites one-for-one.

Default mapped classes:
  - schools: Overture place primary category containing a school token;
  - markets: market/farmers_market/etc., excluding 'supermarket' by default;
  - drinking water: Overture base infrastructure class == drinking_water;
  - toilets: Overture base infrastructure class == toilets.

The exact mapped category counts used are exported for auditability.
"""
from __future__ import annotations

import argparse
import gzip
import os
import json
import math
import pickle
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

PAPER_RELEASE = "2026-08-19.0"
PAPER_SCHEMA_VERSION = "v1.18.0"
STAC_ROOT = "https://stac.overturemaps.org/catalog.json"
STAC_PREFIX = "https://stac.overturemaps.org"
CALENDAR_URL = "https://docs.overturemaps.org/release-calendar.json"
RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")

CHECKPOINT_VERSION = 7
DPI = int(os.environ.get("DOUALA_FIGURE_DPI", "450"))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",default=None)
    p.add_argument("--city_slug",default="douala")
    p.add_argument("--run_id",type=int,default=3000)
    p.add_argument("--places-file",default=None,help="Optional local Overture places GeoParquet/GeoJSON fallback.")
    p.add_argument("--infrastructure-file",default=None,help="Optional local Overture infrastructure GeoParquet/GeoJSON fallback.")
    p.add_argument("--min-place-confidence",type=float,default=0.0,help="Optional confidence filter; default retains all mapped places.")
    p.add_argument("--release", default="latest", help="'latest' discovers the live release via official STAC; or supply YYYY-MM-DD.N for an available pinned release.")
    p.add_argument("--check-release-only", action="store_true", help="Print the selected Overture release/schema without querying Parquet or requiring model checkpoints.")
    p.add_argument("--save-feature-data", action="store_true", help="Opt in to saving raw mapped geometry and assigned feature-level Parquet. Disabled by default; CSV summaries are always saved.")
    return p.parse_args()


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "douala-overture-validation/1.1 (research reproduction)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not retrieve Overture metadata from {url}: {exc}") from exc


def _resolve_release(requested: str) -> dict:
    """STAC is authoritative for 'latest'; release catalogue gives schema version.

    An explicitly pinned release does NOT silently fall back to a newer one.
    """
    if requested == "latest":
        root = _fetch_json(STAC_ROOT)
        release = str(root.get("latest", ""))
        if not RELEASE_RE.fullmatch(release):
            raise RuntimeError(f"Official Overture STAC has no valid latest release: {release!r}")
        discovery = STAC_ROOT
    else:
        if not RELEASE_RE.fullmatch(requested):
            raise ValueError("--release must be 'latest' or YYYY-MM-DD.N (e.g. 2026-10-21.0)")
        release = requested
        discovery = "explicit --release"

    schema = "unknown"
    schema_source = "unavailable"
    try:
        catalog = _fetch_json(f"{STAC_PREFIX}/{release}/catalog.json")
        version = catalog.get("schema:version") or catalog.get("schema_version")
        if version:
            schema = "v" + str(version).lstrip("v")
            schema_source = "official STAC release catalog"
    except RuntimeError as exc:
        # A pinned, expired release must fail on Parquet access; do NOT substitute data.
        print(f"[Overture] release catalogue metadata not accessible: {exc}", file=sys.stderr)

    if schema == "unknown":
        try:
            calendar = _fetch_json(CALENDAR_URL)
            for row in calendar.get("releases", []):
                if row.get("dataVersion") == release and row.get("schemaVersion"):
                    schema = str(row["schemaVersion"])
                    schema_source = "official release calendar"
                    break
        except RuntimeError as exc:
            print(f"[Overture] schema-calendar metadata unavailable: {exc}", file=sys.stderr)

    return {"release": release, "schema_version": schema,
            "schema_version_source": schema_source, "discovery": discovery,
            "places_s3": f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*",
            "infrastructure_s3": f"s3://overturemaps-us-west-2/release/{release}/theme=base/type=infrastructure/*"}


def _remote_columns(con, parquet_path: str) -> set[str]:
    """Read Parquet metadata only; select a field that truly exists in that release."""
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=1)").fetchall()
    except Exception as exc:
        raise RuntimeError(f"Cannot read Overture Parquet metadata at {parquet_path}. "
                           f"The release may be expired or the compute node cannot access public S3: {exc}") from exc
    return {str(row[0]) for row in rows}


def _cloud_sql(bbox, min_conf, places_path, infra_path, place_columns, infra_columns):
    """Only select the cross-version category field; analytical filters remain unchanged."""
    if "categories" in place_columns:  # Schema v1.18.0 category field
        category_expr = "categories.primary"
        category_field = "categories.primary"
    elif "taxonomy" in place_columns:  # September 2026 onwards: v2.0.0
        category_expr = "taxonomy.primary"
        category_field = "taxonomy.primary"
    else:
        raise RuntimeError("Unsupported Overture Places schema: neither categories nor taxonomy exists. "
                           "Stop and review schema changes before accepting results.")
    if not {"id", "geometry", "bbox", "confidence"}.issubset(place_columns):
        raise RuntimeError(f"Unsupported Places schema: missing {sorted({'id','geometry','bbox','confidence'}-place_columns)}")
    if not {"id", "geometry", "bbox", "class"}.issubset(infra_columns):
        raise RuntimeError(f"Unsupported infrastructure schema: missing {sorted({'id','geometry','bbox','class'}-infra_columns)}")
    name_expr = "names.primary" if "names" in place_columns else "NULL"
    basic_expr = "basic_category" if "basic_category" in place_columns else "NULL"
    minx, miny, maxx, maxy = bbox
    place_sql = f"""
        SELECT id,
               {name_expr} AS name,
               {category_expr} AS category,
               {basic_expr} AS basic_category,
               confidence,
               ST_AsWKB(geometry) AS geometry_wkb
        FROM read_parquet('{places_path}', hive_partitioning=1)
        WHERE bbox.xmin <= {maxx} AND bbox.xmax >= {minx}
          AND bbox.ymin <= {maxy} AND bbox.ymax >= {miny}
          AND (confidence IS NULL OR confidence >= {float(min_conf)})
    """
    infra_sql = f"""
        SELECT id,
               "class" AS infra_class,
               ST_AsWKB(ST_Centroid(geometry)) AS geometry_wkb
        FROM read_parquet('{infra_path}', hive_partitioning=1)
        WHERE bbox.xmin <= {maxx} AND bbox.xmax >= {minx}
          AND bbox.ymin <= {maxy} AND bbox.ymax >= {miny}
          AND "class" IN ('drinking_water','toilets')
    """
    return place_sql, infra_sql, category_field


def _style():
    plt.rcParams.update({"font.size":13,"axes.labelsize":14,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11,"savefig.dpi":DPI,"pdf.fonttype":42})


def _load_layout(ckdir:Path):
    p=ckdir/f"v{CHECKPOINT_VERSION}_layout_baseline.pkl.gz"
    if not p.exists(): raise FileNotFoundError(f"Baseline layout checkpoint missing: {p}")
    with gzip.open(p,"rb") as f: return pickle.load(f)


def _study_layers(layout):
    hubs=layout.get("inner_hubs_active")
    if hubs is None or getattr(hubs,"empty",True): raise ValueError("Baseline layout does not contain inner_hubs_active.")
    hubs=hubs.reset_index(drop=True).copy()
    hubs["hub_id"]=np.arange(len(hubs),dtype=int)
    boundary=layout.get("fixed_analysis_zones")
    if boundary is None or getattr(boundary,"empty",True): boundary=layout.get("outer_cells_active")
    if boundary is None or getattr(boundary,"empty",True): boundary=hubs
    boundary_geom=boundary.geometry.union_all()
    return hubs,boundary_geom


def _bbox4326(boundary_geom,crs):
    gs=gpd.GeoSeries([boundary_geom],crs=crs).to_crs(4326)
    minx,miny,maxx,maxy=gs.total_bounds
    return float(minx),float(miny),float(maxx),float(maxy)


def _duckdb_connect():
    try: import duckdb
    except ImportError as exc:
        raise RuntimeError("Python package 'duckdb' is required for direct Overture cloud queries. Install it once in the Conda environment or provide local --places-file and --infrastructure-file inputs.") from exc
    con=duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        # Extensions may already be installed but unavailable to INSTALL on a
        # compute node without internet. Try LOAD separately before failing.
        try: con.execute("LOAD spatial; LOAD httpfs;")
        except Exception as exc:
            raise RuntimeError("DuckDB spatial/httpfs extensions could not be loaded. Run the provided environment/preflight step on a network-enabled login node, or use local Overture extracts.") from exc
    con.execute("SET s3_region='us-west-2'")
    return con


def _from_wkb_frame(df,crs=4326):
    if df.empty: return gpd.GeoDataFrame(df,geometry=[],crs=crs)
    geom=[wkb.loads(bytes(x)) if x is not None else None for x in df.pop("geometry_wkb")]
    return gpd.GeoDataFrame(df,geometry=geom,crs=crs)


def _query_cloud(bbox, min_conf, release_info):
    con = _duckdb_connect()
    try:
        place_columns = _remote_columns(con, release_info["places_s3"])
        infra_columns = _remote_columns(con, release_info["infrastructure_s3"])
        place_sql, infra_sql, category_field = _cloud_sql(
            bbox, min_conf, release_info["places_s3"], release_info["infrastructure_s3"],
            place_columns, infra_columns)
        print(f"[Overture] using primary category field: {category_field}", flush=True)
        print(f"[Overture] querying release {release_info['release']} places...", flush=True)
        places = _from_wkb_frame(con.execute(place_sql).fetchdf())
        print(f"[Overture] places returned in bbox: {len(places):,}", flush=True)
        print(f"[Overture] querying release {release_info['release']} infrastructure...", flush=True)
        infra = _from_wkb_frame(con.execute(infra_sql).fetchdf())
        print(f"[Overture] drinking_water/toilets returned in bbox: {len(infra):,}", flush=True)
    finally:
        con.close()
    return places, infra, category_field


def _read_local(path,kind):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    g=gpd.read_file(p) if p.suffix.lower() in {".geojson",".json",".gpkg",".shp"} else gpd.read_parquet(p)
    if g.crs is None: g=g.set_crs(4326)
    # Accept either already-flattened local fields or common Overture names.
    if kind=="places":
        if "category" not in g.columns:
            for c in ["categories.primary", "categories_primary", "taxonomy.primary", "taxonomy_primary", "basic_category"]:
                if c in g.columns: g["category"] = g[c]; break
        if "category" not in g.columns:
            for nested in ("categories", "taxonomy"):
                if nested in g.columns:
                    g["category"] = g[nested].map(lambda x: x.get("primary") if isinstance(x, dict) else None)
                    break
        if "category" not in g.columns: raise ValueError("Local places file needs category, categories.primary, taxonomy.primary, a nested categories/taxonomy struct, or basic_category.")
        if "name" not in g.columns: g["name"]=""
        if "confidence" not in g.columns: g["confidence"]=np.nan
    else:
        if "infra_class" not in g.columns:
            if "class" in g.columns: g["infra_class"]=g["class"]
            else: raise ValueError("Local infrastructure file needs 'class' or 'infra_class'.")
    return g.to_crs(4326)


def _classify_places(places):
    p=places.copy(); cat=p["category"].fillna("").astype(str).str.lower()
    # School token is intentionally transparent and narrow: 'school' as a
    # standalone underscore-delimited token, plus kindergarten.
    school=cat.str.contains(r"(^|_)school($|_)",regex=True) | cat.eq("kindergarten")
    # Market token deliberately excludes supermarket/hypermarket unless Overture
    # explicitly classifies the place as market/marketplace/farmers_market/etc.
    market=cat.str.contains(r"(^|_)market($|_)",regex=True) | cat.eq("marketplace")
    return p.loc[school].copy(),p.loc[market].copy()


def _filter_boundary(gdf,boundary_geom,target_crs):
    if gdf.empty: return gdf.to_crs(target_crs)
    z=gdf.to_crs(target_crs).copy()
    # Convert any non-point mapped geometry to a representative point before hub assignment.
    z["geometry"]=z.geometry.apply(lambda geom: geom if geom is None or geom.geom_type=="Point" else geom.representative_point())
    return z[z.geometry.intersects(boundary_geom)].copy()


def _assign_hubs(gdf,hubs):
    if gdf.empty: return np.asarray([],dtype=int)
    polys=hubs.reset_index(drop=True); cents=np.column_stack((polys.geometry.centroid.x,polys.geometry.centroid.y))
    try: sidx=polys.sindex
    except Exception: sidx=None
    out=[]
    for pt in gdf.geometry:
        found=-1; cand=list(sidx.intersection(pt.bounds)) if sidx is not None else range(len(polys))
        for i in cand:
            geom=polys.geometry.iloc[int(i)]
            if geom.contains(pt) or geom.touches(pt): found=int(i); break
        if found<0: found=int(np.argmin(np.hypot(cents[:,0]-pt.x,cents[:,1]-pt.y)))
        out.append(found)
    return np.asarray(out,dtype=int)


def _gini(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]; x=np.clip(x,0,None)
    if len(x)==0 or x.sum()<=0:return np.nan
    x=np.sort(x); n=len(x); return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n)


def _dist_metrics(counts):
    x=np.asarray(counts,dtype=float); total=x.sum(); n=len(x)
    if total<=0:return {"count":0,"largest_hub_share_pct":np.nan,"top2_hub_share_pct":np.nan,"hhi":np.nan,"normalised_entropy":np.nan,"gini":np.nan,"dominant_hub":np.nan},np.zeros(n)
    p=x/total; nz=p[p>0]; ent=-np.sum(nz*np.log(nz)); norm=ent/np.log(n) if n>1 else 0.0
    return {"count":int(total),"largest_hub_share_pct":100*float(p.max()),"top2_hub_share_pct":100*float(np.sort(p)[-2:].sum()),"hhi":float(np.sum(p*p)),"normalised_entropy":float(norm),"gini":_gini(x),"dominant_hub":int(np.argmax(p))},p


def _synthetic_layers(layout,hubs):
    out={}
    mapping={"schools":"schools_active","markets":"markets_active","drinking_water":"taps_active","toilets":"toilets_active"}
    for name,key in mapping.items():
        g=layout.get(key)
        if g is None or getattr(g,"empty",True): out[name]=gpd.GeoDataFrame(geometry=[],crs=hubs.crs)
        else: out[name]=g.to_crs(hubs.crs) if g.crs!=hubs.crs else g.copy()
    return out


def _hub_counts(gdf,hubs):
    ids=_assign_hubs(gdf,hubs); counts=np.bincount(ids,minlength=len(hubs)).astype(float) if len(ids) else np.zeros(len(hubs),dtype=float)
    return counts,ids


def main():
    args=parse_args(); _style()
    release_info = _resolve_release(args.release)
    print(f"[Overture] selected release={release_info['release']} schema={release_info['schema_version']} ({release_info['discovery']})", flush=True)
    if args.check_release_only:
        print(json.dumps(release_info, indent=2))
        return
    root=Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    run=root/"outputs"/args.city_slug/f"run_{args.run_id:03d}"/"cooccurrence_ensemble"; ckdir=run/"checkpoints"
    out=run/"overture_validation"; out.mkdir(parents=True,exist_ok=True)
    layout=_load_layout(ckdir); hubs,boundary=_study_layers(layout); bbox=_bbox4326(boundary,hubs.crs)
    print(f"[Overture] study bbox WGS84: {bbox}")

    if args.places_file:
        places=_read_local(args.places_file,"places")
    else:
        places=None
    if args.infrastructure_file:
        infra=_read_local(args.infrastructure_file,"infrastructure")
    else:
        infra=None
    category_field = "local input"
    if places is None or infra is None:
        cloud_places,cloud_infra,category_field=_query_cloud(bbox,args.min_place_confidence,release_info)
        if places is None: places=cloud_places
        if infra is None: infra=cloud_infra

    # Feature-level data is optional, not part of the default publication archive.
    if args.save_feature_data:
        places.to_parquet(out/"overture_places_bbox_raw.parquet",index=False)
        infra.to_parquet(out/"overture_infrastructure_bbox_raw.parquet",index=False)
    category_counts=places.assign(category=places["category"].fillna("<missing>").astype(str)).groupby("category").size().reset_index(name="n").sort_values("n",ascending=False)
    category_counts.to_csv(out/"overture_place_primary_category_counts_in_bbox.csv",index=False)

    schools,markets=_classify_places(places)
    mapped={
        "schools":_filter_boundary(schools,boundary,hubs.crs),
        "markets":_filter_boundary(markets,boundary,hubs.crs),
        "drinking_water":_filter_boundary(infra[infra.infra_class.astype(str).eq("drinking_water")],boundary,hubs.crs),
        "toilets":_filter_boundary(infra[infra.infra_class.astype(str).eq("toilets")],boundary,hubs.crs),
    }
    synth=_synthetic_layers(layout,hubs)
    mapped["named_poi"]=pd.concat([mapped["schools"],mapped["markets"]],ignore_index=True)
    mapped["named_poi"]=gpd.GeoDataFrame(mapped["named_poi"],geometry="geometry",crs=hubs.crs)
    mapped["wash"]=pd.concat([mapped["drinking_water"],mapped["toilets"]],ignore_index=True)
    mapped["wash"]=gpd.GeoDataFrame(mapped["wash"],geometry="geometry",crs=hubs.crs)
    synth["named_poi"]=pd.concat([synth["schools"],synth["markets"]],ignore_index=True); synth["named_poi"]=gpd.GeoDataFrame(synth["named_poi"],geometry="geometry",crs=hubs.crs)
    synth["wash"]=pd.concat([synth["drinking_water"],synth["toilets"]],ignore_index=True); synth["wash"]=gpd.GeoDataFrame(synth["wash"],geometry="geometry",crs=hubs.crs)

    count_rows=[]; share_rows=[]; metric_rows=[]
    for category in ["schools","markets","drinking_water","toilets","named_poi","wash"]:
        mc,mids=_hub_counts(mapped[category],hubs); sc,sids=_hub_counts(synth[category],hubs)
        mm,mp=_dist_metrics(mc); sm,sp=_dist_metrics(sc)
        count_rows.append({"category":category,"mapped_count":int(mc.sum()),"synthetic_count":int(sc.sum()),"mapped_to_synthetic_count_ratio":float(mc.sum()/sc.sum()) if sc.sum()>0 else np.nan})
        for hub_id in range(len(hubs)):
            share_rows.append({"category":category,"hub_id":hub_id,"mapped_count":int(mc[hub_id]),"synthetic_count":int(sc[hub_id]),"mapped_share_pct":100*mp[hub_id] if mp.sum()>0 else np.nan,"synthetic_share_pct":100*sp[hub_id] if sp.sum()>0 else np.nan})
        row={"category":category}
        for prefix,m in [("mapped",mm),("synthetic",sm)]:
            for k,v in m.items():row[f"{prefix}_{k}"]=v
        if mc.sum()>0 and sc.sum()>0:
            js=float(jensenshannon(mp,sp,base=2.0))
            mad=float(np.mean(np.abs(mp-sp))*100.0); maxad=float(np.max(np.abs(mp-sp))*100.0); overlap=float(np.minimum(mp,sp).sum()*100.0)
            rho,pv=spearmanr(mp,sp) if len(mp)>=3 and np.std(mp)>0 and np.std(sp)>0 else (np.nan,np.nan)
            row.update({
                "synthetic_minus_mapped_largest_hub_share_pp":float(sm["largest_hub_share_pct"]-mm["largest_hub_share_pct"]),
                "absolute_largest_hub_share_difference_pp":abs(float(sm["largest_hub_share_pct"]-mm["largest_hub_share_pct"])),
                "hhi_difference_synthetic_minus_mapped":float(sm["hhi"]-mm["hhi"]),
                "normalised_entropy_difference_synthetic_minus_mapped":float(sm["normalised_entropy"]-mm["normalised_entropy"]),
                "jensen_shannon_distance":js,
                "mean_absolute_hub_share_difference_pp":mad,
                "max_absolute_hub_share_difference_pp":maxad,
                "distribution_overlap_pct":overlap,
                "spearman_hub_share_rho":float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p":float(pv) if np.isfinite(pv) else np.nan,
                "same_dominant_hub":bool(mm["dominant_hub"]==sm["dominant_hub"]),
            })
        metric_rows.append(row)

        # Save auditable assigned mapped features.
        if args.save_feature_data:
            z=mapped[category].copy(); z["hub_id"]=_assign_hubs(z,hubs) if len(z) else []
            if len(z): z.to_parquet(out/f"mapped_{category}_assigned_to_model_hubs.parquet",index=False)

    counts=pd.DataFrame(count_rows); shares=pd.DataFrame(share_rows); metrics=pd.DataFrame(metric_rows)
    counts.to_csv(out/"baseline_validation_feature_counts.csv",index=False)
    shares.to_csv(out/"baseline_validation_hub_shares.csv",index=False)
    metrics.to_csv(out/"baseline_validation_metrics.csv",index=False)

    # Classification audit: which Overture place categories entered schools/markets.
    audit=[]
    for label,g in [("schools",schools),("markets",markets)]:
        vc=g["category"].fillna("<missing>").astype(str).value_counts()
        for category,n in vc.items():audit.append({"validation_class":label,"overture_primary_category":category,"n_in_bbox_before_boundary_filter":int(n)})
    pd.DataFrame(audit).to_csv(out/"baseline_validation_place_categories_used.csv",index=False)

    # Figure 1: direct concentration comparison.
    order=["schools","markets","drinking_water","toilets","named_poi","wash"]
    z=metrics.set_index("category").reindex(order); y=np.arange(len(order)); fig,ax=plt.subplots(figsize=(10.8,7.2));
    ax.plot(z["mapped_largest_hub_share_pct"],y,"o",label="Overture mapped"); ax.plot(z["synthetic_largest_hub_share_pct"],y,"s",label="Synthetic Baseline")
    for i in range(len(z)):
        a=z["mapped_largest_hub_share_pct"].iloc[i]; b=z["synthetic_largest_hub_share_pct"].iloc[i]
        if np.isfinite(a) and np.isfinite(b): ax.plot([a,b],[i,i],linewidth=1.2,alpha=.5)
    ax.set_yticks(y); ax.set_yticklabels([x.replace("_"," ").title() for x in order]); ax.invert_yaxis(); ax.set_xlabel("Largest-hub share (%)"); ax.grid(axis="x",alpha=.22); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"baseline_validation_largest_hub_share.png",dpi=DPI,bbox_inches="tight"); fig.savefig(out/"baseline_validation_largest_hub_share.pdf",bbox_inches="tight"); plt.close(fig)

    # Figure 2: distribution-similarity metrics (lower JS/MAD = closer; overlap higher = closer).
    fig,ax=plt.subplots(figsize=(10.8,7.2)); ax.scatter(z["jensen_shannon_distance"],y,s=65); ax.set_yticks(y); ax.set_yticklabels([x.replace("_"," ").title() for x in order]); ax.invert_yaxis(); ax.set_xlabel("Jensen-Shannon distance between hub-share distributions (0 = identical)"); ax.set_xlim(left=0); ax.grid(axis="x",alpha=.22); fig.tight_layout(); fig.savefig(out/"baseline_validation_distribution_distance.png",dpi=DPI,bbox_inches="tight"); fig.savefig(out/"baseline_validation_distribution_distance.pdf",bbox_inches="tight"); plt.close(fig)

    # One combined mapped-vs-synthetic hub-share scatter per category, separate files for readability.
    for category in order:
        q=shares[shares.category.eq(category)].dropna(subset=["mapped_share_pct","synthetic_share_pct"])
        fig,ax=plt.subplots(figsize=(7.8,7.2)); ax.scatter(q.mapped_share_pct,q.synthetic_share_pct,s=55); lim=max(1.0,float(np.nanmax(q[["mapped_share_pct","synthetic_share_pct"]].to_numpy())) if len(q) else 1.0); ax.plot([0,lim],[0,lim],linestyle="--",linewidth=1); ax.set_xlim(0,lim*1.05); ax.set_ylim(0,lim*1.05); ax.set_xlabel("Overture mapped hub share (%)"); ax.set_ylabel("Synthetic Baseline hub share (%)"); ax.set_aspect("equal",adjustable="box"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f"baseline_validation_hub_share_scatter_{category}.png",dpi=DPI,bbox_inches="tight"); fig.savefig(out/f"baseline_validation_hub_share_scatter_{category}.pdf",bbox_inches="tight"); plt.close(fig)

    metadata={
        "overture_release":release_info["release"], "overture_schema_version":release_info["schema_version"],
        "overture_schema_version_source":release_info["schema_version_source"],
        "release_discovery":release_info["discovery"],
        "publication_reference_release":PAPER_RELEASE,
        "publication_reference_schema":PAPER_SCHEMA_VERSION,
        "is_publication_reference_release":release_info["release"] == PAPER_RELEASE,
        "places_category_field":category_field,
        "validation_utc":datetime.now(timezone.utc).isoformat(),
        "save_feature_data":bool(args.save_feature_data),
        "places_source":str(args.places_file) if args.places_file else release_info["places_s3"],
        "infrastructure_source":str(args.infrastructure_file) if args.infrastructure_file else release_info["infrastructure_s3"],
        "study_bbox_wgs84":bbox,"n_model_hubs":len(hubs),"min_place_confidence":args.min_place_confidence,
        "school_rule":"primary category contains underscore-delimited 'school' token, or equals kindergarten",
        "market_rule":"primary category contains underscore-delimited 'market' token, or equals marketplace; supermarket is not included solely because its string contains market",
        "wash_rule":"base infrastructure class exactly drinking_water or toilets",
        "interpretation":"External spatial plausibility check only. Mapped coverage may be incomplete or spatially uneven; mapped total counts are descriptive and are not treated as ground truth facility totals.",
    }
    (out/"baseline_validation_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # Keep a release-specific copy of small aggregate tables, preventing silent loss
    # of older validation summaries if a different 'latest' release is run later.
    snap=out/"release_summaries"/release_info["release"]
    snap.mkdir(parents=True,exist_ok=True)
    for filename in ["baseline_validation_feature_counts.csv", "baseline_validation_hub_shares.csv", "baseline_validation_metrics.csv", "baseline_validation_place_categories_used.csv", "baseline_validation_metadata.json"]:
        (snap/filename).write_bytes((out/filename).read_bytes())

    lines=[f"Overture Baseline plausibility check — release {release_info['release']}, schema {release_info['schema_version']}",
           f"Reference release: {PAPER_RELEASE} ({PAPER_SCHEMA_VERSION}); selected release: {release_info['release']}.",
           "Mapped data are an external plausibility check, not complete ground truth.", ""]
    for _,r in metrics.iterrows():
        lines.append(f"{r['category']}: mapped n={int(r['mapped_count'])}, synthetic n={int(r['synthetic_count'])}; largest-hub mapped={r['mapped_largest_hub_share_pct']:.2f}% vs synthetic={r['synthetic_largest_hub_share_pct']:.2f}% (synthetic−mapped={r.get('synthetic_minus_mapped_largest_hub_share_pp',np.nan):.2f} pp); JS distance={r.get('jensen_shannon_distance',np.nan):.3f}; overlap={r.get('distribution_overlap_pct',np.nan):.1f}%.")
    lines += ["", "Interpret mapped category counts and concentration statistics with attention to mapped coverage."]
    (out/"baseline_validation_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"[Overture] outputs → {out}")
    print("[Overture] SUCCESS")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"[Overture] FAILED: {exc}",file=sys.stderr); raise
