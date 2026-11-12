# Agent-based modelling of spatial co-occurrence and cholera exposure under urban decentralisation

This repository contains the Python implementation and Slurm workflow for a spatially explicit agent-based model of Douala, Cameroon. It compares four synthetic configurations: Baseline, POI redistribution, WASH redistribution, and combined POI+WASH redistribution. The model examines mobility, shared-infrastructure use, contamination and simulated exposure; it is not calibrated to forecast a particular outbreak.

## Files

- `code/`: model (`douala_model.py`), ensemble driver, statistical analysis, checkpoint and figure validation, and the Overture Maps spatial assessment
- `slurm/`: numbered HPC jobs for precomputation, 50-seed simulations, aggregation, figures, Overture and results packaging
- `configs/`: example city configuration
- `inputs/`: required WorldPop input formats and provenance notes (data are not distributed)
- `environment/`: Conda dependency export
- `install_to_aire.sh` and `submit_douala_pipeline.sh`: deployment and Slurm submission

## Inputs and environment

The workflow requires the prepared WorldPop 2020 Cameroon population raster and CSV at `inputs/douala/data/pop.tif` and `inputs/douala/data/pop.csv`, plus access to the OpenStreetMap street network. See `inputs/README.md` and the accompanying article for data sources. The exact map geometry may depend on the externally obtained network extract.

The model uses Python 3.11 with the packages listed in `environment/environment.yml`. The supplied Slurm scripts target the Leeds Aire `nodes` partition, the `airenotebook` Conda environment and `/mnt/scratch/$USER/cholera_project`. Adapt paths, modules and job resources for other HPC systems.

## Execution on Aire

Place the repository in the project directory, ensure `configs/cities.csv` and the prepared population input files exist, and run:

```bash
cd /mnt/scratch/$USER/cholera_project
bash douala-cholera-spatial-model/install_to_aire.sh
bash submit_douala_pipeline.sh
squeue -u "$USER"
```

The workflow performs input preflight and spatial precomputation, runs seeds 42–91 for all four scenarios using a Slurm array, validates and aggregates the 200 scenario outputs, generates statistics and figures, performs Overture validation, and packages non-checkpoint results as a `.tgz`. The output directory is `outputs/douala/run_3000/`. Before submitting, ensure this run directory does not contain checkpoints from another run; checkpoint resumption is enabled.

## Overture Maps

The external spatial plausibility assessment reported in the article uses Overture Maps release `2026-08-19.0`. The validator can select an available release using `--release` and defaults to the latest available release from the official STAC catalogue. The selected release and schema are saved with the results. Mapped results depend on the source release and do not affect the agent-based simulations.

```bash
python code/validate_overture_baseline.py --check-release-only --release latest
```

## Outputs

The main output directory contains per-seed data, scenario summaries, `statistics/`, `figures/`, `overture_validation/`, validation reports and a compressed results archive. Simulation settings, definitions and analysis procedures are provided in the code and accompanying manuscript.
