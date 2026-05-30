"""
One-time: extract the project's FSA polygons to a small GeoJSON the frontend
can read WITHOUT geopandas.

Run in the env that has geopandas (Person 1's repo .venv), once:

    .venv/bin/python frontend/export_fsa_geojson.py

Output: artifacts/toronto_fsa.geojson  (only the ~96 FSAs in world, EPSG:4326).
"""
import json
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
SHP = ROOT / "datasets/fsa_boundaries/lfsa000b21a_e/lfsa000b21a_e.shp"
META = ROOT / "datasets/world_meta.json"
OUT = ROOT / "artifacts/toronto_fsa.geojson"


def main():
    fsa_codes = set(json.loads(META.read_text())["fsa_index"])
    g = gpd.read_file(SHP)
    g = g[g["CFSAUID"].isin(fsa_codes)].copy()
    g = g.to_crs(epsg=4326)                       # web-map coords
    g = g[["CFSAUID", "geometry"]].rename(columns={"CFSAUID": "fsa"})
    # simplify a touch to keep the file small/snappy (tolerance in degrees)
    g["geometry"] = g["geometry"].simplify(0.0002, preserve_topology=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(OUT, driver="GeoJSON")
    print(f"wrote {OUT}  ({len(g)}/{len(fsa_codes)} FSAs, {OUT.stat().st_size//1024} KB)")
    missing = fsa_codes - set(g["fsa"])
    if missing:
        print(f"WARNING: {len(missing)} FSA codes not found in shapefile: {sorted(missing)}")


if __name__ == "__main__":
    main()
