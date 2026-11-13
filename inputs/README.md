# Spatial inputs

The population data are not distributed with this code. Download the Cameroon
**2020 unconstrained WorldPop population raster (30 arc-second resolution)**,
corresponding to the source specified in Supplementary Table S2:
https://hub.worldpop.org/geodata/summary?id=30553

Place the extracted GeoTIFF at:

    inputs/douala/data/pop.tif

Convert it to the point/weight CSV used by the model:

    python code/prepare_population.py \
      --raster inputs/douala/data/pop.tif \
      --csv inputs/douala/data/pop.csv

The CSV has WGS84 longitude (`x`), latitude (`y`), and population value (`z`)
for each positive-value raster cell. The conversion is deterministic. Do not
substitute a different population product if you expect the same spatial
geometry as the study.

The street network is obtained from OpenStreetMap as part of the spatial
precomputation. A newly retrieved network can differ from the historical
network extract used for the article. The Overture Maps validator downloads
**independent validation features**; Overture data never enter the epidemic
simulation or the figure inputs.
