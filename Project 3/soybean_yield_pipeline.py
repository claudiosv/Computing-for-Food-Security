#!/usr/bin/env python


import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import urllib
import urllib.parse
import urllib.request
from time import sleep
from urllib.error import HTTPError
import os
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import requests


from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from c_usda_quick_stats import c_usda_quick_stats

MY_NASS_API_key = "A269B59D-8921-3BAB-B00A-26507C5E9D29"


def curr_timestamp():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime


def get_coordinate_pixel(tiff_file, lon, lat):
    dataset = rasterio.open(tiff_file)
    py, px = dataset.index(lon, lat)

    window = rasterio.windows.Window(px, py, 1, 1)

    clip = dataset.read(window=window)

    return clip[0][0][0]


output_dir = Path("USDA-NASS--v01/OUTPUTS/")

archive_dir = Path("ML-ARCHIVES--v01/")

tif_dir = Path("GAEZ-SOIL-for-ML/OUTPUTS/")

weather_dir = archive_dir / "WEATHER-DATA--v01/"

ml_tables_dir = archive_dir / "ML-TABLES--v01/"

output_dir.mkdir(parents=True, exist_ok=True)
tif_dir.mkdir(parents=True, exist_ok=True)
weather_dir.mkdir(parents=True, exist_ok=True)
ml_tables_dir.mkdir(parents=True, exist_ok=True)


farm_survey_1997_file = output_dir / "national_farm_survey_acres_ge_1997.csv"

if not farm_survey_1997_file.exists():
    parameters = (
        "source_desc=SURVEY"
        + "&"
        + urllib.parse.quote("sector_desc=FARMS & LANDS & ASSETS")
        + "&"
        + urllib.parse.quote("commodity_desc=FARM OPERATIONS")
        + "&"
        + urllib.parse.quote("statisticcat_desc=AREA OPERATED")
        + "&unit_desc=ACRES"
        + "&freq_desc=ANNUAL"
        + "&reference_period_desc=YEAR"
        + "&year__GE=1997"
        + "&agg_level_desc=NATIONAL"
        + "&"
        + urllib.parse.quote("state_name=US TOTAL")
        + "&format=CSV"
    )

    stats = c_usda_quick_stats(MY_NASS_API_key)
    s_json = stats.get_data(parameters, farm_survey_1997_file)
else:
    print("Skipping national_farm_survey_acres_ge_1997")

soybean_yield_data = output_dir / "soybean_yield_data_raw.csv"


if not soybean_yield_data.exists():
    parameters = (
        "source_desc=SURVEY"
        + "&sector_desc=CROPS"
        + "&"
        + urllib.parse.quote("group_desc=FIELD CROPS")
        + "&commodity_desc=SOYBEANS"
        + "&statisticcat_desc=YIELD"
        + "&geographic_level=STATE"
        + "&agg_level_desc=COUNTY"
        + "&state_name=ILLINOIS"
        + "&state_name=IOWA"
        + "&state_name=MINNESTOTA"
        + "&state_name=INDIANA"
        + "&state_name=OHIO"
        + "&state_name=NEBRASKA"
        + "&state_name=MISSOURI"
        + "&year__GE=2003"
        + "&year__LE=2022"
        + "&format=CSV"
    )

    stats = c_usda_quick_stats(MY_NASS_API_key)
    stats.get_data(parameters, soybean_yield_data)
else:
    print("skipping soybean_yield_data")

tgt_file = archive_dir / "soybean_yield_data.csv"
if not tgt_file.exists():
    df = pd.read_csv(soybean_yield_data)

    df1 = df[["short_desc"]].drop_duplicates()
    print(df1.head(10))

    df = df[df["short_desc"] == "SOYBEANS - YIELD, MEASURED IN BU / ACRE"]
    print(len(df))

    bad_county_names = ["OTHER COUNTIES", "OTHER (COMBINED) COUNTIES"]
    df = df[~df.county_name.isin(bad_county_names)]

    print(len(df))

    df2 = df[["state_name", "county_name"]].drop_duplicates()
    print(len(df2))

    df = df.rename(columns={"Value": "yield"})

    output_file = output_dir / "repaired_yield.csv"

    df.to_csv(output_file, index=False)

    shutil.copyfile(output_file, tgt_file)
else:
    print("not copying ", tgt_file)

tgt_file_01 = archive_dir / "year_state_county_yield.csv"
if not tgt_file_01.exists():
    tgt_file = archive_dir / "soybean_yield_data.csv"

    df = pd.read_csv(tgt_file)

    cols_to_keep = ["year", "state_name", "county_name", "yield"]
    dfml = df[cols_to_keep]

    print(dfml.head())

    print(dfml.shape[0])

    print(dfml[dfml["yield"].isnull()].head())

    dfml.to_csv(tgt_file_01, index=False)
    print("\nwrote file ", tgt_file_01)
else:
    print("not writing ", tgt_file_01)

state_county_lon_lat = archive_dir / "state_county_lon_lat.csv"
if not state_county_lon_lat.exists():
    file = archive_dir / "year_state_county_yield.csv"

    df = pd.read_csv(file)
    print("number of rows in csv cleaned for ML: ", len(df))

    print(df.head())

    df1 = df[["state_name", "county_name"]].drop_duplicates()
    print("\nNumber of state-county pairs is: ", len(df1))

    index = df.index[
        (df["county_name"] == "DU PAGE") | (df["county_name"] == "DUPAGE")
    ].tolist()
    for ind in index:
        df.at[ind, "county_name"] = "DUPAGE"

    index1 = df1.index[
        (df1["county_name"] == "DU PAGE") | (df1["county_name"] == "DUPAGE")
    ].tolist()
    for ind in index1:
        df1.at[ind, "county_name"] = "DUPAGE"

    def geocode_county(state, county):
        geolocator = Nominatim(user_agent="county_geocoder")
        location = geolocator.geocode(county + ", " + state + ", USA")
        sleep(0.5)
        if location:
            return location.longitude, location.latitude
        else:
            print("no lat-lon found for ", state, county)
            return None, None

    df1[["lon", "lat"]] = df1.apply(
        lambda x: geocode_county(x["state_name"], x["county_name"]),
        axis=1,
        result_type="expand",
    )

    print(df1.head())

    print("lon-lat for ILLINOIS-BUREAU is: ", geocode_county("ILLINOIS", "BUREAU"))

    df1.to_csv(state_county_lon_lat, index=False)
    print("wrote file: ", state_county_lon_lat)
else:
    print("not writing ", state_county_lon_lat)


state_county_lon_lat_soil = archive_dir / "state_county_lon_lat_soil.csv"
if not state_county_lon_lat_soil.exists():
    scll_filename = archive_dir / "state_county_lon_lat.csv"
    df_scll = pd.read_csv(scll_filename)
    print(df_scll.head())
    print(len(df_scll))
    urlkeys = {
        "AEZ_classes": "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/aez/aez_v9v2red_5m_CRUTS32_Hist_8110_100_avg.tif",
        "nutr_ret_high": "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ2_mze_v9aH.tif",
        "soil_qual_high": "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ0_mze_v9aH.tif",
        "soil_qual_low": "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ0_mze_v9aL.tif",
        "suit_irrig_high_soy": "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res05/CRUTS32/Hist/8110H/suHi_soy.tif",
    }

    fileFullName = {}

    for key, url in urlkeys.items():
        tif_file = tif_dir / f"{key}.tif"
        if not tif_file.exists():
            fileFullName[key] = tif_file
            print(fileFullName[key])
            urllib.request.urlretrieve(url, tif_file)
        else:
            print("not retrieving ", key)

    def pull_useful(
        ginfo,
    ):
        useful = {}
        useful["band_count"] = len(ginfo["bands"])

        useful["size"] = ginfo["size"]

        return useful

    gdalInfoReq = {}
    gdalInfo = {}
    useful = {}
    for k in urlkeys.keys():
        gdalInfoReq[k] = " ".join(["gdalinfo", "-json", fileFullName[k]])

        result = subprocess.run(
            [gdalInfoReq[k]], shell=True, capture_output=True, text=True
        )
        gdalInfo[k] = json.loads(result.stdout)

        useful[k] = pull_useful(gdalInfo[k])
        print("\n", k)
        print(json.dumps(useful[k], indent=2, sort_keys=True))

    tiff_file = fileFullName["AEZ_classes"]

    print(df_scll.iloc[[0]])
    test_lon = df_scll.iloc[0]["lon"]
    test_lat = df_scll.iloc[0]["lat"]
    print(test_lon, test_lat, type(test_lon), type(test_lat))
    val = get_coordinate_pixel(tiff_file, test_lon, test_lat)
    print(type(val))
    print(val)

    df3 = df_scll.copy()
    print(df3.head())
    print(len(df3))
    for k in urlkeys.keys():
        tiff_file = fileFullName[k]
        df3[k] = df3.apply(
            lambda r: get_coordinate_pixel(tiff_file, r["lon"], r["lat"]), axis=1
        )
    print(df3.head())
    print(len(df3))

    for k in urlkeys.keys():
        print(k)
        print(df3[[k]].drop_duplicates().head(100))

    df4 = df3.copy()

    one_hot = pd.get_dummies(df4["AEZ_classes"])

    df4 = df4.drop("AEZ_classes", axis=1)

    df4 = df4.join(one_hot)
    print(len(df4))
    print(df4.head())
    print(df4.columns.tolist())

    cols = {
        16: "AEZ_1",
        17: "AEZ_2",
        18: "AEZ_3",
        19: "AEZ_4",
        20: "AEZ_5",
        21: "AEZ_6",
        27: "AEZ_7",
        28: "AEZ_8",
        32: "AEZ_9",
    }
    df4 = df4.rename(columns=cols)
    print(df4.columns.tolist())
    print(df4.head())

    df5 = df4.copy()

    one_hot1 = pd.get_dummies(df5["soil_qual_high"])

    df5 = df5.drop("soil_qual_high", axis=1)

    df5 = df5.join(one_hot1)
    print(len(df5))
    print(df5.head())
    print(df5.columns.tolist())

    cols = {
        4: "SQH_1",
        5: "SQH_2",
        6: "SQH_3",
        7: "SQH_4",
        8: "SQH_5",
        9: "SQH_6",
        10: "SQH_7",
    }
    df5 = df5.rename(columns=cols)
    print(df5.columns.tolist())
    print(df5.head())

    df6 = df5.copy()

    one_hot2 = pd.get_dummies(df6["soil_qual_low"])

    df6 = df6.drop("soil_qual_low", axis=1)

    df6 = df6.join(one_hot2)
    print(len(df6))
    print(df6.head())
    print(df6.columns.tolist())

    cols = {
        4: "SQL_1",
        5: "SQL_2",
        6: "SQL_3",
        7: "SQL_4",
        8: "SQL_5",
        9: "SQL_6",
        10: "SQL_7",
    }
    df6 = df6.rename(columns=cols)
    print(df6.columns.tolist())
    print(df6.head())
    state_county_lon_lat_soil = archive_dir / "state_county_lon_lat_soil.csv"
    df6.to_csv(state_county_lon_lat_soil, index=False)
    print("wrote file: ", state_county_lon_lat_soil)
else:
    print("Skipping state_county_lon_lat_soil")


yscy_file = archive_dir / "year_state_county_yield.csv"
scll_file = archive_dir / "state_county_lon_lat.csv"
df_yscy = pd.read_csv(yscy_file)
df_scll = pd.read_csv(scll_file)


index_list = df_yscy.index[
    (df_yscy["county_name"] == "DU PAGE") | (df_yscy["county_name"] == "DUPAGE")
].tolist()
print(index_list)
for i in index_list:
    df_yscy.at[i, "county_name"] = "DUPAGE"
    print(df_yscy.at[i, "county_name"])
print(len(df_yscy), len(df_scll))
df_yscyll = pd.merge(df_yscy, df_scll, on=["state_name", "county_name"], how="left")
print(len(df_yscyll))


print(df_yscyll[df_yscyll["year"] == 2022].head(10))
print(df_scll.head(10))

print(df_yscyll.iloc[279:284].head())
yscyll_filename = archive_dir / "year_state_county_yield_lon_lat.csv"
df_yscyll.to_csv(yscyll_filename, index=False)
print("wrote file: ", yscyll_filename)
yscyll_filename = archive_dir / "year_state_county_yield_lon_lat.csv"
df_yscyll = pd.read_csv(yscyll_filename)

weather_params = [
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "GWETROOT",
    "EVPTRNS",
    "ALLSKY_SFC_PAR_TOT",
]
"""
   T2M_MAX: The maximum hourly air (dry bulb) temperature at 2 meters above the surface of the
             earth in the period of interest.
   T2M_MIN: The minimum hourly air (dry bulb) temperature at 2 meters above the surface of the
            earth in the period of interest.
   PRECTOTCORR: The bias corrected average of total precipitation at the surface of the earth
                in water mass (includes water content in snow)
   EVPTRNS: The evapotranspiration energy flux at the surface of the earth
   ALLSKY_SFC_PAR_TOT: The total Photosynthetically Active Radiation (PAR) incident
         on a horizontal plane at the surface of the earth under all sky conditions
"""


base_url = r"https://power.larc.nasa.gov/api/temporal/daily/point?"
base_url += (
    "parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,GWETROOT,EVPTRNS,ALLSKY_SFC_PAR_TOT&"
)
base_url += "community=RE&longitude={longitude}&latitude={latitude}&start={year}0401&end={year}1031&format=JSON"


def fetch_weather_county_year(year, state, county):
    row = df_yscyll.loc[
        (df_yscyll["state_name"] == state)
        & (df_yscyll["county_name"] == county)
        & (df_yscyll["year"] == year)
    ]

    lon = row.iloc[0]["lon"]
    lat = row.iloc[0]["lat"]

    api_request_url = base_url.format(longitude=lon, latitude=lat, year=str(year))

    response = requests.get(url=api_request_url, verify=True, timeout=30.00)

    content = json.loads(response.content.decode("utf-8"))

    weather = content["properties"]["parameter"]

    df = pd.DataFrame(weather)
    return df


df = fetch_weather_county_year(2022, "ILLINOIS", "LEE")


print(len(df))

print(df.head())


w_df = {}


out_dir = archive_dir / "WEATHER-DATA--v01/"
filename = r"weather-data-for-index__{index}.csv"

starttime = datetime.datetime.now().strftime("%Y-%m-% %H:%M:%S")

for i in range(0, len(df_yscyll)):
    row = df_yscyll.iloc[i]
    outfilename = out_dir + filename.format(index=str(i).zfill(4))

    if Path(outfilename).exists():
        continue

    w_df[i] = fetch_weather_county_year(
        row["year"], row["state_name"], row["county_name"]
    )
    w_df[i].to_csv(outfilename)

    if i % 10 == 0:
        print(
            "\nFinished work on index: ",
            i,
            "     at time: ",
            datetime.datetime.now().strftime("%Y-%m-% %H:%M:%S"),
        )
        print("   This involved fetching weather data for the following row:")
        print(
            row["year"], row["state_name"], row["county_name"], row["lon"], row["lat"]
        )
        print("Wrote file: ", outfilename)


endtime = datetime.datetime.now().strftime("%Y-%m-% %H:%M:%S")
print("start and end times were: ", starttime, endtime)

yscyll_filename = "year_state_county_yield_lon_lat.csv"

wdtemplate = r"weather-data-for-index__{padded}.csv"

df_yscyll = pd.read_csv(archive_dir / yscyll_filename)
print(df_yscyll.shape)


w_df = {}
for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    w_df[i] = pd.read_csv(weather_dir / wdtemplate.format(padded=padded))
    w_df[i].rename(columns={"Unnamed: 0": "date"}, inplace=True)


print(w_df[4].shape)
print(w_df[4].head())


def create_monthly_df(df):
    df1 = df.copy()

    df1.index = pd.to_datetime(df["date"], format="%Y%m%d")

    df1_monthly = df1.resample("M").agg(
        {
            "T2M_MAX": "mean",
            "T2M_MIN": "mean",
            "PRECTOTCORR": "sum",
            "GWETROOT": "mean",
            "EVPTRNS": "mean",
            "ALLSKY_SFC_PAR_TOT": "sum",
        }
    )

    df1_monthly.index = df1_monthly.index.strftime("%Y%m%d")

    return df1_monthly


print(create_monthly_df(w_df[4]).head(50))


df_t0 = w_df[0]
cols_narrow = df_t0.columns.values.tolist()[1:]
print(cols_narrow)


df_t1 = create_monthly_df(df_t0)
print(len(df_t1))


cols_wide = []
for i in range(0, len(df_t1)):
    row = df_t1.iloc[i]

    month_id = "month_" + str(i).zfill(2)

    for c in cols_narrow:
        cols_wide.append(month_id + "__" + c)

print(cols_wide)
print(len(cols_wide))


print(w_df[0].columns.tolist()[1:])
print(w_df[0].shape)
print(create_monthly_df(w_df[0]).shape)


def create_weather_seq_for_monthly(dfw):
    seq = []
    cols = dfw.columns.tolist()
    for i in range(0, len(dfw)):
        for c in cols:
            seq.append(dfw.iloc[i][c])
    return seq


dfw = create_monthly_df(w_df[0])
print(dfw.head(10))

seqw = create_weather_seq_for_monthly(dfw)
print(json.dumps(seqw, indent=4))


u_df = {}
dfw = {}
seqw = {}


for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)

    u_df[padded] = pd.read_csv(weather_dir + wdtemplate.format(padded=padded))

    u_df[padded].rename(columns={"Unnamed: 0": "date"}, inplace=True)

    dfw[padded] = create_monthly_df(u_df[padded])

    seqw[i] = create_weather_seq_for_monthly(dfw[padded])


print(len(seqw))
print(json.dumps(seqw, indent=4))


print(dfw["0000"].shape)
print(len(cols_wide))
print(len(df_yscyll))
print(len(seqw[0]))


df_wide_weather_monthly = pd.DataFrame.from_dict(
    seqw, orient="index", columns=cols_wide
)

print(df_wide_weather_monthly.shape)

print(df_wide_weather_monthly.head())


sclls_file = archive_dir / "state_county_lon_lat_soil.csv"

df_scsoil = pd.read_csv(sclls_file).drop(columns=["lon", "lat"])
print(df_scsoil.shape)


df_ysc_y_soil = pd.merge(
    df_yscyll, df_scsoil, on=["state_name", "county_name"], how="left"
)

df_ysc_y_soil = df_ysc_y_soil.drop(columns=["lon", "lat"])


print(df_ysc_y_soil.shape)
print(df_ysc_y_soil.head())


df_ysc_y_soil_weather_monthly = pd.concat(
    [df_ysc_y_soil, df_wide_weather_monthly], axis="columns"
)

print(df_ysc_y_soil_weather_monthly.shape)

print(df_ysc_y_soil_weather_monthly.loc[28:32, :])


ml_file = ml_tables_dir / "ML-table-monthly.csv"

df_ysc_y_soil_weather_monthly.to_csv(ml_file, index=False)

print("Wrote file ", ml_file)


yscyll_filename = archive_dir / "year_state_county_yield_lon_lat.csv"

wdtemplate = r"weather-data-for-index__{padded}.csv"

df_yscyll = pd.read_csv(yscyll_filename)
print(df_yscyll.shape)


w_df = {}
for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    w_df[i] = pd.read_csv(weather_dir / wdtemplate.format(padded=padded))

    w_df[i].rename(columns={"Unnamed: 0": "date"}, inplace=True)


print(w_df[4].shape)
print(w_df[4].head())


def create_weekly_df(df):
    df1 = df.copy()

    df1.index = pd.to_datetime(df["date"], format="%Y%m%d")

    df1_weekly = df1.resample("W").agg(
        {
            "T2M_MAX": "mean",
            "T2M_MIN": "mean",
            "PRECTOTCORR": "sum",
            "GWETROOT": "mean",
            "EVPTRNS": "mean",
            "ALLSKY_SFC_PAR_TOT": "sum",
        }
    )

    df1_weekly.index = df1_weekly.index.strftime("%Y%m%d")

    return df1_weekly


print(create_weekly_df(w_df[4]).head(50))


df_t0 = w_df[0]
cols_narrow = df_t0.columns.values.tolist()[1:]
print(cols_narrow)


df_t1 = create_weekly_df(df_t0)


cols_wide = []
for i in range(0, len(df_t1)):
    row = df_t1.iloc[i]

    week_id = "week_" + str(i).zfill(2)

    for c in cols_narrow:
        cols_wide.append(week_id + "__" + c)

print(cols_wide)


print(w_df[0].columns.tolist()[1:])
print(w_df[0].shape)
print(create_weekly_df(w_df[0]).shape)


def create_weather_seq_for_weekly(dfw):
    seq = []
    for i in range(0, len(dfw)):
        for c in cols:
            seq.append(dfw.iloc[i][c])
    return seq


dfw = create_weekly_df(w_df[0])
print(dfw.head(10))

seqw = create_weather_seq_for_weekly(dfw)
print(json.dumps(seqw, indent=4))


u_df = {}
dfw = {}
seqw = {}


for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)

    u_df[padded] = pd.read_csv(weather_dir / wdtemplate.format(padded=padded))

    u_df[padded].rename(columns={"Unnamed: 0": "date"}, inplace=True)

    dfw[padded] = create_weekly_df(u_df[padded])

    seqw[i] = create_weather_seq_for_weekly(dfw[padded])

    if i % 100 == 0:
        print("Completed processing of index ", i)


print(print(json.dumps(seqw, indent=4)))


print(len(seqw))


print(dfw["0000"].shape)
print(len(cols_wide))
print(len(df_yscyll))
print(len(seqw[0]))


df_wide_weather_weekly_prelim = pd.DataFrame.from_dict(
    seqw, orient="index", columns=cols_wide
)

print(df_wide_weather_weekly_prelim.shape)

print(df_wide_weather_weekly_prelim.head())


print(df_wide_weather_weekly_prelim.shape)
week_31_cols = [
    "week_31__T2M_MAX",
    "week_31__T2M_MIN",
    "week_31__PRECTOTCORR",
    "week_31__GWETROOT",
    "week_31__EVPTRNS",
    "week_31__ALLSKY_SFC_PAR_TOT",
]

df_wide_weather_weekly = df_wide_weather_weekly_prelim.drop(columns=week_31_cols)


print(df_wide_weather_weekly.shape)
print(df_wide_weather_weekly.head())


sclls_file = archive_dir / "state_county_lon_lat_soil.csv"

df_scsoil = pd.read_csv(sclls_file).drop(columns=["lon", "lat"])
print(df_scsoil.shape)


df_ysc_y_soil = pd.merge(
    df_yscyll, df_scsoil, on=["state_name", "county_name"], how="left"
)

df_ysc_y_soil = df_ysc_y_soil.drop(columns=["lon", "lat"])


print(df_ysc_y_soil.shape)
print(df_ysc_y_soil.head())


df_ysc_y_soil_weather_weekly = pd.concat(
    [df_ysc_y_soil, df_wide_weather_weekly], axis="columns"
)

print(df_ysc_y_soil_weather_weekly.shape)

print(df_ysc_y_soil_weather_weekly.loc[28:32, :])


ml_file = ml_tables_dir / "ML-table-weekly.csv"

df_ysc_y_soil_weather_weekly.to_csv(ml_file, index=False)

print("Wrote file ", ml_file)


ml_tables_dir = archive_dir + "ML-TABLES--v01/"

ml_file = "ML-table-monthly.csv"


df_ml = pd.read_csv(ml_tables_dir + ml_file)
print(df_ml.shape)
print(df_ml.head())


print(df_ml.isnull().values.any())


X = df_ml.drop(columns=["yield"])
y = df_ml.loc[:, ["yield"]]

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


y_test_orig = y_test.copy()

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

print(y_test_orig.head())


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_test.iloc[0, 0])


X_train = X_train.drop(columns=["year", "state_name", "county_name"])
X_test = X_test.drop(columns=["year", "state_name", "county_name"])


scalerXST = StandardScaler().fit(X_train)
scaleryST = StandardScaler().fit(y_train)

X_trainST = scalerXST.transform(X_train)
y_trainST = scaleryST.transform(y_train)
X_testST = scalerXST.transform(X_test)
y_testST = scaleryST.transform(y_test)


print(df_ml.iloc[7397]["yield"])
print(y_testST[0])
print(scaleryST.inverse_transform(y_testST)[0])


lassoST = Lasso(alpha=0.1)
lassoST.fit(X_trainST, y_trainST)


y_predST = lassoST.predict(X_testST)

print(y_predST)


rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))

print(rmseST)


def plot_predictions(y_test, y_pred, descrip_of_run):
    if len(y_test) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="red", linestyle="--")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Predicted vs Actual for " + descrip_of_run)
    plt.grid(True)
    plt.show()


plot_predictions(y_testST, y_predST, "Lasso with StandardScalar")


linearST = LinearRegression()


print(type(y_trainST))

linearST.fit(X_trainST, y_trainST)


y_predST = linearST.predict(X_testST)

rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))
print(rmseST)

plot_predictions(y_testST, y_predST, "Linear Regression using StandardScaler")


scalerXMM = MinMaxScaler().fit(X_train)
scaleryMM = MinMaxScaler().fit(y_train)

X_trainMM = scalerXMM.transform(X_train)
y_trainMM = scaleryMM.transform(y_train)
X_testMM = scalerXMM.transform(X_test)
y_testMM = scaleryMM.transform(y_test)


print(df_ml.iloc[7397]["yield"])
print(y_testMM[0])
print(scaleryMM.inverse_transform(y_testMM)[0])


linearMM = LinearRegression()

linearMM.fit(X_trainMM, y_trainMM)

y_predMM = linearMM.predict(X_testMM)

rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Linear Regression using MinMaxScaler")


regrMM = RandomForestRegressor(max_depth=2, random_state=0)


regrMM.fit(X_trainMM, y_trainMM.ravel())

y_predMM = regrMM.predict(X_testMM)
rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Random Forest Regressor using MinMaxScaler")


#!/usr/bin/env python


ml_tables_dir = archive_dir + "ML-TABLES--v01/"

ml_file = "ML-table-weekly.csv"


df_ml = pd.read_csv(ml_tables_dir + ml_file)
print(df_ml.shape)
print(df_ml.head())


print(df_ml.isnull().values.any())


X = df_ml.drop(columns=["yield"])
y = df_ml.loc[:, ["yield"]]

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


y_test_orig = y_test.copy()

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

print(y_test_orig.head())


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_test.iloc[0, 0])


X_train = X_train.drop(columns=["year", "state_name", "county_name"])
X_test = X_test.drop(columns=["year", "state_name", "county_name"])


scalerXST = StandardScaler().fit(X_train)
scaleryST = StandardScaler().fit(y_train)

X_trainST = scalerXST.transform(X_train)
y_trainST = scaleryST.transform(y_train)
X_testST = scalerXST.transform(X_test)
y_testST = scaleryST.transform(y_test)


print(df_ml.iloc[7397]["yield"])
print(y_testST[0])
print(scaleryST.inverse_transform(y_testST)[0])


lassoST = Lasso(alpha=0.1)
lassoST.fit(X_trainST, y_trainST)


y_predST = lassoST.predict(X_testST)

print(y_predST)


rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))

print(rmseST)


def plot_predictions(y_test, y_pred, descrip_of_run):
    if len(y_test) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="red", linestyle="--")
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Predicted vs Actual for " + descrip_of_run)
    plt.grid(True)
    plt.show()


plot_predictions(y_testST, y_predST, "Lasso with StandardScalar")


linearST = LinearRegression()


print(type(y_trainST))

linearST.fit(X_trainST, y_trainST)


y_predST = linearST.predict(X_testST)

rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))
print(rmseST)

plot_predictions(y_testST, y_predST, "Linear Regression using StandardScaler")


scalerXMM = MinMaxScaler().fit(X_train)
scaleryMM = MinMaxScaler().fit(y_train)

X_trainMM = scalerXMM.transform(X_train)
y_trainMM = scaleryMM.transform(y_train)
X_testMM = scalerXMM.transform(X_test)
y_testMM = scaleryMM.transform(y_test)


print(df_ml.iloc[7397]["yield"])
print(y_testMM[0])
print(scaleryMM.inverse_transform(y_testMM)[0])


linearMM = LinearRegression()

linearMM.fit(X_trainMM, y_trainMM)

y_predMM = linearMM.predict(X_testMM)

rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Linear Regression using MinMaxScaler")


regrMM = RandomForestRegressor(max_depth=20, random_state=0)


regrMM.fit(X_trainMM, y_trainMM.ravel())

y_predMM = regrMM.predict(X_testMM)
rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Random Forest Regressor using MinMaxScaler")
