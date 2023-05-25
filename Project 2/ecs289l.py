# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# mamba create -n geo python=3.11 jupyterlab pandas numpy localtileserver ipyleaflet rasterio scipy geopandas shapely matplotlib rioxarray voila
# mamba activate geo
# # if on a mac
# mamba install numpy "libblas=*=*accelerate"

import glob
from pathlib import Path
from urllib import request
from voila import getCoordinatePixel
import pandas as pd
import rasterio
import rioxarray as rxr
from rasterio.warp import transform
from tqdm.auto import tqdm


def get_tif_corners(file_path):
    with rasterio.open(file_path) as src:
        # Get the bounds of the raster
        left, bottom, right, top = src.bounds

        # Get the CRS (Coordinate Reference System) of the raster
        src_crs = src.crs

        # Transform the corners from the source CRS to WGS84 (EPSG:4326)
        top_left_lon, top_left_lat = transform(src_crs, "EPSG:4326", [left], [top])
        bottom_right_lon, bottom_right_lat = transform(
            src_crs, "EPSG:4326", [right], [bottom]
        )

        # Return the corners in WGS84
        return (top_left_lat[0], top_left_lon[0]), (
            bottom_right_lat[0],
            bottom_right_lon[0],
        )


# actual yield
water = [
    "T",  # total
    "R",  # rainfed
    "I",  # irrigated
]
time_periods = ["2000", "2010"]
variable = [
    "yld",  # yield
    "prd",  # production
    "har",  # harvested area
]

# production
crops = [
    "ban",  # banana
    "ricd",  # dryland rice
    "lmze",  # lowland maize
    "maiz",  # maize
    "ricw",  # wetland rice
    "whea",  # wheat
    "wwhe",  # winter wheat
]
url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res06/{water}/{time_periods}/{crops}_{time_periods}_{variable}.tif"

# crop value actual
crops = [
    "all",  # main
    "cer",  # cereal
    "oil",  # oil seeds
    "rts",  # root crops
]
# -

# # value in (1000 GK$)
# "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res06/V/2000/all_2000_val.tif" # main
# "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res06/V/2000/cer_2000_val.tif" # cereal
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res06/V/2000/oil_2000_val.tif # oil seeds
#             https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res06/V/2010/rts_2010_val.tif # root crops
#
# # potential
# 2020: 2011-2040
# 2050: 2041-2070
# 2080: 2071-2100
#
#
# # agro
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res02/GFDL-ESM2M/rcp8p5/2020sH/whea200a_yld.tif # irrigation 2011-2040
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res02/GFDL-ESM2M/rcp8p5/2020sH/whea200b_yld.tif # rain

rcps = ["rcp2p6", "rcp8p5"]
years = ["2020", "2050", "2080"]
crops = [
    "bana",  # banana
    "ricd",  # dryland rice
    "lmze",  # lowland maize
    "maiz",  # maize
    "ricw",  # wetland rice
    "whea",  # wheat
    "wwhe",  # winter wheat
    "soyb",  # soybean
    "sugc",
]
waters = [
    "a",  # irrigation
    "b",  # rain
]
for year in years:
    for rcp in rcps:
        for crop in crops:
            for water in waters:
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res02/GFDL-ESM2M/{rcp}/{year}sH/{crop}200{water}_yld.tif"
                filename = (
                    f"potential_yield_gfdlesm2m_{rcp}_{year}_{crop}200{water}_yld.tif"
                )
                request.urlretrieve(url, filename)

# +
# Agro-climatic
# Thermal Regime
# Mean annual temp
# 2071-2100 -> 2080s
# 2011-2040 -> 2020s
# 2041-2070 -> 2050s
# Climate model ENSEMBLE
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/ENSEMBLE/rcp2p6/tmp_ENSEMBLE_rcp2p6_2080s.tif
# year 2080
# model GFDL-ESM2M
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp2p6/TS/tmp_GFDL-ESM2M_rcp2p6_2080.tif
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/TS/tmp_GFDL-ESM2M_rcp8p5_2092.tif
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/TS/tmp_GFDL-ESM2M_rcp8p5_2054.tif
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/tmp_GFDL-ESM2M_rcp8p5_2054.tif
# model HadGEM2-ES
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/HadGEM2-ES/rcp2p6/TS/tmp_HadGEM2-ES_rcp2p6_2080.tif
# model IPSL-CM5A-LR
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/IPSL-CM5A-LR/rcp2p6/TS/tmp_IPSL-CM5A-LR_rcp2p6_2080.tif
# MIROC-ESM-CHEM
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/MIROC-ESM-CHEM/rcp2p6/TS/tmp_MIROC-ESM-CHEM_rcp2p6_2080.tif
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/NorESM1-M/rcp2p6/TS/tmp_NorESM1-M_rcp2p6_2080.tif


# annual precipitation mm
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/NorESM1-M/rcp8p5/TS/prc_NorESM1-M_rcp8p5_2080.tif

# total number rain days
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/TS/ndr_GFDL-ESM2M_rcp8p5_2054.tif

# +
rcps = ["rcp2p6", "rcp8p5"]
years = ["2020s", "2050s", "2080s"] + list(map(str, range(2011, 2100)))
models = ["ENSEMBLE", "GFDL-ESM2M"]

for year in tqdm(years):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = f"Agroclimatic/Thermal regime/Mean annual temp/ts_tmp_{model}_{rcp}_{year}.tif"
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/TS/tmp_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)

rcps = ["rcp2p6", "rcp8p5"]
years = ["2020s", "2050s", "2080s"] + list(map(str, range(2011, 2100)))
models = ["GFDL-ESM2M"]

for year in tqdm(years):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = f"Agroclimatic/Moisture regime/Annual precipitation/prc_{model}_{rcp}_{year}.tif"
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/TS/prc_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)

# +
rcps = ["rcp2p6", "rcp8p5"]
year_range = ["2020s", "2050s", "2080s"]
models = ["GFDL-ESM2M"]

for year in tqdm(year_range):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = f"Agroclimatic/Moisture regime/Annual precipitation/prc_{model}_{rcp}_{year}.tif"
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/prc_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)

# +
# Number of rain days
rcps = ["rcp2p6", "rcp8p5"]
years = list(map(str, range(2011, 2100)))
models = ["GFDL-ESM2M"]
year_range = ["2020s", "2050s", "2080s"]

for year in tqdm(year_range):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = f"Agroclimatic/Moisture regime/Number of rain days/ndr_{model}_{rcp}_{year}.tif"
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/ndr_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)
for year in tqdm(years):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = f"Agroclimatic/Moisture regime/Number of rain days/ndr_{model}_{rcp}_{year}.tif"
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/TS/ndr_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)

# Tmax > 35 and Tmin < 15
rcps = ["rcp2p6", "rcp8p5"]
years = list(map(str, range(2011, 2100)))
models = ["GFDL-ESM2M"]  # "HadGEM2-ES",  ]
year_range = ["2020s", "2050s", "2080s"]

for year in tqdm(year_range):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = (
                f"Agroclimatic/Thermal regime/Tmin_lt_15/n15_{model}_{rcp}_{year}.tif"
            )
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/n15_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)
for year in tqdm(years):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = (
                f"Agroclimatic/Thermal regime/Tmin_lt_15/n15_{model}_{rcp}_{year}.tif"
            )
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/TS/n15_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)
for year in tqdm(year_range):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = (
                f"Agroclimatic/Thermal regime/Tmax_gt_35/x35_{model}_{rcp}_{year}.tif"
            )
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/x35_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)
for year in tqdm(years):
    for rcp in tqdm(rcps):
        for model in tqdm(models):
            filename = (
                f"Agroclimatic/Thermal regime/Tmax_gt_35/x35_{model}_{rcp}_{year}.tif"
            )
            file = Path(filename)
            if not file.exists():
                url = f"https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/{model}/{rcp}/TS/x35_{model}_{rcp}_{year}.tif"
                try:
                    request.urlretrieve(url, filename)
                except:
                    print("Could not fetch", url)

long = -80.1151
lat = -3.7957
years = list(map(str, range(2011, 2100)))
records = []
record = {
    "year": "",
    "rcp": "",
    "precipitation_mm": "",
    "rain_days": "",
    "mean_temp": "",
    "tmax35": "",
    "tmin15": "",
}
for year in years:
    for rcp in rcps:
        record = {
            "year": year,
            "rcp": rcp,
            "precipitation_mm": getCoordinatePixel(
                f"Agroclimatic/Moisture regime/Annual precipitation/prc_GFDL-ESM2M_{rcp}_{year}.tif",
                long,
                lat,
            ),
            "rain_days": getCoordinatePixel(
                f"Agroclimatic/Moisture regime/Number of rain days/ndr_GFDL-ESM2M_{rcp}_{year}.tif",
                long,
                lat,
            ),
            "mean_temp": getCoordinatePixel(
                f"Agroclimatic/Thermal regime/Mean annual temp/ts_tmp_GFDL-ESM2M_{rcp}_{year}.tif",
                long,
                lat,
            ),
            "tmax35": getCoordinatePixel(
                f"Agroclimatic/Thermal regime/Tmax_gt_35/x35_GFDL-ESM2M_{rcp}_{year}.tif",
                long,
                lat,
            ),
            "tmin15": getCoordinatePixel(
                f"Agroclimatic/Thermal regime/Tmin_lt_15/n15_GFDL-ESM2M_{rcp}_{year}.tif",
                long,
                lat,
            ),
        }
        records.append(record)

df = pd.DataFrame.from_records(records)
df.to_csv(f"hfdl-esm2m-{lat}_-{long}.csv")

df[["year", "rcp", "mean_temp"]].set_index("year").groupby("rcp")["mean_temp"].plot(
    legend=True,
    xlabel="Year",
    ylabel="Mean Temperature °C",
    title="Mean Temperature per Year",
)

df[["year", "rcp", "precipitation_mm"]].set_index("year").groupby("rcp")[
    "precipitation_mm"
].plot(
    legend=True,
    xlabel="Year",
    ylabel="Annual precipitation (mm)",
    title="Annual precipitation",
)

df[["year", "rcp", "rain_days"]].set_index("year").groupby("rcp")["rain_days"].plot(
    legend=True,
    xlabel="Year",
    ylabel="Number of rain days",
    title="Number of rain days per year",
)

df[["year", "rcp", "tmax35"]].set_index("year").groupby("rcp")["tmax35"].plot(
    legend=True,
    xlabel="Year",
    ylabel="Number of days $T_{max}$ > 35°C",
    title="Number of days $T_{max}$ > 35°C",
)

df[["year", "rcp", "tmin15"]].set_index("year").groupby("rcp")["tmin15"].plot(
    legend=True,
    xlabel="Year",
    ylabel="Number of days $T_{min}$ < 15°C",
    title="Number of days $T_{min}$ < 15°C",
)

# +
# tmin < 15
##https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/n15_GFDL-ESM2M_rcp8p5_2050s.tif

# tmax > 35
# https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res01/GFDL-ESM2M/rcp8p5/x35_GFDL-ESM2M_rcp8p5_2050s.tif
# -

# # List tif metadata

# +
# Get a list of all .tif files in the current directory
tif_files = glob.glob("*.tif")

# Apply the function to each file
for file in tif_files:
    raster = rxr.open_rasterio(file, masked=True)

    # print("The datatype of raster is:", type(raster))

    print(
        file, "The Coordinate Reference System (CRS) of your data is:", raster.rio.crs
    )
    print("\nThe bounds of your data are:", raster.rio.bounds())
    print("\nThe shape of your data is:", raster.shape)
    print("\nThe spatial resolution for your data is:", raster.rio.resolution())
    print("\nThe metadata for your data is:", raster.attrs)
    print("\nThe nodatavalue of your data is:", raster.rio.nodata)
