#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook is fetching soybean yields for an ML pipeline </spann>
#
# <span style=color:blue>It pulls from USDA NASS.</span>
#


# This useful if I want to give unique names to directories or files
# This useful if I want to give unique names to directories or files
# This useful if I want to give unique names to directories or files
# This useful if I want to give unique names to directories or files
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
import xml.etree.ElementTree as ET
from time import sleep
from urllib.error import HTTPError
import os
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import requests

# import seaborn as sns
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def curr_timestamp():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime


# ### <span style=color:blue> Accessing USDA NASS, following code from https://towardsdatascience.com/harvest-and-analyze-agricultural-data-with-the-usda-nass-api-python-and-tableau-a6af374b8138.  In first cell below we define a class for interacting with the NASS QuickStats API, and in second cell we illustrate how to invoke that class </span>


# from https://towardsdatascience.com/harvest-and-analyze-agricultural-data-with-the-usda-nass-api-python-and-tableau-a6af374b8138
# with edits

#   Name:           c_usda_quick_stats.py
#   Author:         Randy Runtsch
#   Date:           March 29, 2022
#   Project:        Query USDA QuickStats API
#   Author:         Randall P. Runtsch
#
#   Description:    Query the USDA QuickStats api_GET API with a specified set of
#                   parameters. Write the retrieved data, in CSV format, to a file.
#
#   See Quick Stats (NASS) API user guide:  https://quickstats.nass.usda.gov/api
#   Request a QuickStats API key here:      https://quickstats.nass.usda.gov/api#param_define
#
#   Attribution: This product uses the NASS API but is not endorsed or certified by NASS.
#
#   Changes
#


# One has to get a NASS API key - please get your own
my_NASS_API_key = "A269B59D-8921-3BAB-B00A-26507C5E9D29"


class c_usda_quick_stats:
    def __init__(self):
        # Set the USDA QuickStats API key, API base URL, and output file path where CSV files will be written.

        # self.api_key = 'PASTE_YOUR_API_KEY_HERE'
        self.api_key = my_NASS_API_key

        self.base_url_api_get = (
            "http://quickstats.nass.usda.gov/api/api_GET/?key=" + self.api_key + "&"
        )

    def get_data(self, parameters, file_path, file_name):
        # Call the api_GET api with the specified parameters.
        # Write the CSV data to the specified output file.

        # Create the full URL and retrieve the data from the Quick Stats server.

        full_url = self.base_url_api_get + parameters
        print(full_url)

        try:
            s_result = urllib.request.urlopen(full_url)
            # print(type(s_result))
            print(s_result.status, s_result.reason)
            # print(s_result.status_code)
            s_text = s_result.read().decode("utf-8")

            # Create the output file and write the CSV data records to the file.

            s_file_name = file_path + file_name
            o_file = open(s_file_name, "w", encoding="utf8")
            o_file.write(s_text)
            o_file.close()
        except HTTPError as error:
            print(error.code, error.reason)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching the data: {e}")
        except ValueError as e:
            print(f"Failed to parse the response data: {e}")
        except:
            print(
                f"Failed because of unknown exception; perhaps the USDA NASS site is down"
            )


# <span style=color:blue>First, a test query based on Randall Runtsch...    </span>


# from https://towardsdatascience.com/harvest-and-analyze-agricultural-data-with-the-usda-nass-api-python-and-tableau-a6af374b8138
# with edits

#   Date:           March 29, 2022
#   Project:        Program controller to query USDA QuickStats API
#   Author:         Randall P. Runtsch
#
#   Description:    Create an instance of the c_usda_quick_stats class. Call it with
#                   the desired search parameter and output file name.
#
#   Attribution: This product uses the NASS API but is not endorsed or certified by NASS.
#
#   Changes
#


output_dir = "USDA-NASS--v01/OUTPUTS/"
archive_dir = "ML-ARCHIVES--v01/"

# Create a string with search parameters, then create an instance of
# the c_usda_quick_stats class and use that to fetch data from QuickStats
# and write it to a file


# the QuickStats site is very senstivite to how the full URL is built up.
# For example, the following spec for the parameters works
# But if you replace the line "'&unit_desc=ACRES' + \" with
# the line "'&' + urllib.parse.quote('unit_desc-ACRES')"
# then the site responds saying that you have exceeded the 50,000 record limit for one query

if False:
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

    stats = c_usda_quick_stats()

    s_json = stats.get_data(
        parameters,
        output_dir,
        "national_farm_survey_acres_ge_1997_" + curr_timestamp() + ".csv",
    )

    # <span style=color:blue>Now a query that fetches useful soybean yield data.  I am focused on the top 7 soy-producing states in the US, and on the years 2003 to 2022.   </span>

    # Create a string with search parameters, then create an instance of
    # the c_usda_quick_stats class and use that to fetch data from QuickStats
    # and write it to a file

    # It took a while to get the parameter names just right...
    #   The parameters names are listed in
    #      https://quickstats.nass.usda.gov/param_define
    #   (some additional resources in https://quickstats.nass.usda.gov/tutorials)
    #   Also, look at the column names that show up in the csv files that you get back
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

    stats = c_usda_quick_stats()

    # holding this timestamp; we may used it to import the created csv file
    latest_curr_timestamp = curr_timestamp()
    filename = "soybean_yield_data__" + latest_curr_timestamp + ".csv"

    stats.get_data(
        parameters, output_dir, "soybean_yield_data__" + latest_curr_timestamp + ".csv"
    )

    # ### <span style=color:blue>After inspecting the output we see that there is double counting.  In particular, see the columns for "short_desc".  So, we will drop all records with short_desc != "SOYBEANS - YIELD, MEASURED IN BU / ACRE"</span>

    df = pd.read_csv(output_dir + filename)
    # print(df.head())

    df1 = df[["short_desc"]].drop_duplicates()
    print(df1.head(10))

    # keep only records about full yield
    df = df[df["short_desc"] == "SOYBEANS - YIELD, MEASURED IN BU / ACRE"]
    print(len(df))
    # 10295

    # found some bad_county_names by visual inspection of the csv
    bad_county_names = ["OTHER COUNTIES", "OTHER (COMBINED) COUNTIES"]
    df = df[~df.county_name.isin(bad_county_names)]

    print(len(df))
    # 9952

    df2 = df[["state_name", "county_name"]].drop_duplicates()
    print(len(df2))
    # 559

    # Note: using SQL I found that of the 559 state-county pairs total:
    #          212 state-county pairs have data for all 20 years
    #          347 state-county pairs have data for < 20 years
    #
    #          486 have year 2022
    #          418 have year 2021
    #          514 have year 2020
    # I will live with that

    # cleaning up a column name
    df = df.rename(columns={"Value": "yield"})

    output_file = "repaired_yield__" + curr_timestamp() + ".csv"

    df.to_csv(output_dir + output_file, index=False)

    # I imported this table into postgres so that I could use SQL ...

    # #### <span style=color:blue>Saving the csv I'm happy with in a designated place in my "archives" directory</span>

    output_dir = "USDA-NASS--v01/OUTPUTS/"
    archives_dir = "ML-ARCHIVES--v01/"
    src_file = output_file  # from preceding cell
    tgt_file = "soybean_yield_data.csv"

    shutil.copyfile(output_dir + src_file, archives_dir + tgt_file)

    # #### <span style=color:blue>Projecting out the columns and records that I don't need for my ML learning table, and archiving that result, also. </span>

    tgt_file = "soybean_yield_data.csv"

    df = pd.read_csv(archives_dir + tgt_file)
    # print(df.head())

    cols_to_keep = ["year", "state_name", "county_name", "yield"]
    dfml = df[cols_to_keep]

    print(dfml.head())

    print(dfml.shape[0])
    # Note: this particular df has 9952 rows

    # checking there are no null values for 'yield':
    print(dfml[dfml["yield"].isnull()].head())

    tgt_file_01 = "year_state_county_yield.csv"
    dfml.to_csv(archives_dir + tgt_file_01, index=False)
    print("\nwrote file ", archives_dir + tgt_file_01)

    #!/usr/bin/env python
    # coding: utf-8

    # ## <span style=color:blue>Fetching the more-or-less central lat lon for each county/state pair of interest in our ML pipeline    </span>
    #

    def curr_timestamp():
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        return formatted_datetime

    # <span style=color:blue>The year_state_county_yield.csv file holds all of the year-county-state triples of interest along with total soybean yield.</span>

    archives_dir = "ML-ARCHIVES--v01/"
    file = "year_state_county_yield.csv"

    df = pd.read_csv(archives_dir + file)
    print("number of rows in csv cleaned for ML: ", len(df))

    print(df.head())

    df1 = df[["state_name", "county_name"]].drop_duplicates()
    print("\nNumber of state-county pairs is: ", len(df1))

    # <span style=color:blue>The function geocode_county defined below won't work on "DU PAGE" county in Illinois.  But it does work on "DUPAGE".  So, changing the name in both df and df1 </span>

    index = df.index[
        (df["county_name"] == "DU PAGE") | (df["county_name"] == "DUPAGE")
    ].tolist()
    print(index)
    for ind in index:
        df.at[ind, "county_name"] = "DUPAGE"
        print(df.at[ind, "county_name"])

    index1 = df1.index[
        (df1["county_name"] == "DU PAGE") | (df1["county_name"] == "DUPAGE")
    ].tolist()
    print(index1)
    for ind in index1:
        df1.at[ind, "county_name"] = "DUPAGE"
        print(df1.at[ind, "county_name"])

    # Geocoding function to retrieve coordinates for a county
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
    # df1['lat'] = df1.apply(lambda x: geocode_county(x['state_name'], x['county_name'])[1], axis=1)

    print(df1.head())

    print("lon-lat for ILLINOIS-BUREAU is: ", geocode_county("ILLINOIS", "BUREAU"))

    # <span style=color:blue>Archiving df1 for later use </span>

    archives_dir = "ML-ARCHIVES--v01/"
    filename = "state_county_lon_lat.csv"
    df1.to_csv(archives_dir + filename, index=False)
    print("wrote file: ", archives_dir + filename)



    #!/usr/bin/env python
    # coding: utf-8
    # ## <span style=color:blue>Fetching GAEZ soil data for an ML pipeline</span>
    #
    # ###  <span style=color:blue>Note: rather than using GAEZ, you might want to use the ISRIC soil data. </span>
    #
    # <span style=color:blue>E.g., see https://soilgrids.org and https://www.isric.org/explore/soilgrids/faq-soilgrids.  (It may take some digging around to make things work.  You probably want to use the EPSG:4326 (Plate Carre) projection.) Also, if you are working with yield data at the county level, then you might want to use soil grid data at the 5k x 5k gridsize, rather than then 1km x 1km or 250m x 250m gridsize. </span>
    def curr_timestamp():
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        return formatted_datetime


    # ### <span style=color:blue>Fetching the file state_county_lon_lat.csv which will be used below</span>
    scll_filename = "state_county_lon_lat.csv"
    df_scll = pd.read_csv(archive_dir + scll_filename)
    print(df_scll.head())
    print(len(df_scll))
    # ### <span style=color:blue>Fetching several .tif files using urllib.</span>
    #
    # <span style=color:blue>I selected this by visual inspection of various GAEZ data sets.  I was looking for data based on soil that appeared to differentiate parts of my region of interest.</span>
    #
    # ### <span style=color:blue>Note: for the 1st, 3rd and 4th tif files fetched below, the pixel values are categorical.  So we will have to use one-hot encodings for these</span>
    tif_dir = "GAEZ-SOIL-for-ML/OUTPUTS/"
    os.makedirs(tif_dir, exist_ok=True)
    url = {}
    # using Land and Water Resources / Dominant AEZ class (33 classes) at 5 arc-minutes
    # Based on 33 AEZ classes, even though pixel values are integer
    url[
        "AEZ_classes"
    ] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/aez/aez_v9v2red_5m_CRUTS32_Hist_8110_100_avg.tif"
    # Using the URL of TIF file Soil Resources / Nutrient retention capacity, high inputs
    # Based on 1 to 10, corresponding to bands 0.0 to 0.1; 0.1 to 02; etc.  So basically a numeric parameter
    url[
        "nutr_ret_high"
    ] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ2_mze_v9aH.tif"
    # using Land and Water Resources / Soil Resources / Most limiting soil quality rating factor, high inputs
    # Based on 11 soil categories (and water), even though pixel values are integer
    url[
        "soil_qual_high"
    ] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ0_mze_v9aH.tif"
    # using Land and Water Resources / Soil Resources / Most limiting soil quality rating factor, low inputs
    # same as previous
    url[
        "soil_qual_low"
    ] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi1/SQ0_mze_v9aL.tif"
    # Leaving this one out because it is 43200 x 21600 pixels; don't want to work with different size input for now...
    # using Land and Water Resources / Soil suitability, rain-fed, low-inputs
    # url['soil_suit_rain_low'] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/LR/soi2/siLr_sss_mze.tif"
    # using Suitability and Attainable Yield / Suitability Index / Suitability index range (0-10000);
    #   within this chose crop = soybean; time period = 1981 to 2010; others empty
    # this has numeric values from 0 to 10,000
    url[
        "suit_irrig_high_soy"
    ] = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res05/CRUTS32/Hist/8110H/suHi_soy.tif"
    urlkeys = url.keys()
    print(urlkeys)
    # Fetch the TIF files using the associated URLs
    curr = curr_timestamp()
    fileFullName = {}


    # fetching the tif files from web and writing into local files
    for k in urlkeys:
        fileFullName[k] = tif_dir + curr + "__" + k + ".tif"
        print(fileFullName[k])
        urllib.request.urlretrieve(url[k], fileFullName[k])
    # ### <span style=color:blue>Fetching meta-data about the .tif files using GDAL, specifically the command-line operator gdalinfo.</span>
    def pull_useful(
        ginfo,
    ):  # should give as input the result.stdout from calling gdalinfo -json
        useful = {}
        useful["band_count"] = len(ginfo["bands"])
        # useful['cornerCoordinates'] = ginfo['cornerCoordinates']
        # useful['proj:transform'] = ginfo['stac']['proj:transform']
        useful["size"] = ginfo["size"]
        # useful['bbox'] = ginfo['stac']['proj:projjson']['bbox']
        # useful['espgEncoding'] = ginfo['stac']['proj:epsg']
        return useful


    gdalInfoReq = {}
    gdalInfo = {}
    useful = {}
    for k in urlkeys:
        gdalInfoReq[k] = " ".join(["gdalinfo", "-json", fileFullName[k]])
        # print(gdalInfoReq[k])
        result = subprocess.run(
            [gdalInfoReq[k]], shell=True, capture_output=True, text=True
        )
        gdalInfo[k] = json.loads(result.stdout)
        # if k == 'AEZ_classes':
        #     print(json.dumps(gdalInfo[k], indent=2, sort_keys=True))
        useful[k] = pull_useful(gdalInfo[k])
        print("\n", k)
        print(json.dumps(useful[k], indent=2, sort_keys=True))


    # <span style=color:blue>Function to pull value from a pixel.  (Thanks to Claudio Spiess)   </span>
    # following https://gis.stackexchange.com/a/299791, adapted to fetch just one pixel
    def get_coordinate_pixel(tiff_file, lon, lat):
        dataset = rasterio.open(tiff_file)
        py, px = dataset.index(lon, lat)
        # create 1x1px window of the pixel
        window = rasterio.windows.Window(px, py, 1, 1)
        # read rgb values of the window
        clip = dataset.read(window=window)
        # print(clip)
        return clip[0][0][0]


    # testing the function
    tiff_file = fileFullName["AEZ_classes"]
    # tiff_file = 'GAEZ-SOIL-for-ML/OUTPUTS/2023-05-20_23-09-36__AEZ_classes.tif'
    print(df_scll.iloc[[0]])
    test_lon = df_scll.iloc[0]["lon"]
    test_lat = df_scll.iloc[0]["lat"]
    print(test_lon, test_lat, type(test_lon), type(test_lat))
    val = get_coordinate_pixel(tiff_file, test_lon, test_lat)
    print(type(val))
    print(val)
    # ## <span style=color:blue>Now adding all 5 soil values to the rows of df_scll.  This takes a while to run. </span>
    #
    # <span style=color:blue>With the rasterio function, the retrieved values are of type int.  If using the gdalinfo-based fucntion, then the values coming from the XML are all strings, one should convert to int when loading into the new df that we are building</span>
    df3 = df_scll.copy()
    print(df3.head())
    print(len(df3))
    for k in urlkeys:
        #     df3[k] = df3.apply(lambda r: fetch_tif_value(r['lon'], r['lat'], k, False), axis=1)
        tiff_file = fileFullName[k]
        df3[k] = df3.apply(
            lambda r: get_coordinate_pixel(tiff_file, r["lon"], r["lat"]), axis=1
        )
    print(df3.head())
    print(len(df3))
    # <span style=color:blue>Looking at full set of values for each of the soil attributes.</span>
    for k in urlkeys:
        print(k)
        print(df3[[k]].drop_duplicates().head(100))
    # ## <span style=color:blue>Replacing the columns for 'AEZ_classes', 'soil_qual_high', 'soil_qual_low' with multiple "1-hot" columns.   (We could also use the OneHotEncoder from scikit, but I'm choosing to do it here and now on the raw data.)</span>
    #
    # <span style=color:blue>Following https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python</span>
    #
    #
    df4 = df3.copy()
    # Get one hot encoding of columns 'AEZ-classes'
    one_hot = pd.get_dummies(df4["AEZ_classes"])
    # Drop original as it is now encoded
    df4 = df4.drop("AEZ_classes", axis=1)
    # Join the encoded df
    df4 = df4.join(one_hot)
    print(len(df4))
    print(df4.head())
    print(df4.columns.tolist())
    # output was ['state_name', 'county_name', 'lon', 'lat', 'nutr_ret_high',
    #             'soil_qual_high', 'soil_qual_low', 'suit_irrig_high_soy',
    #              16, 17, 18, 19, 20, 21, 27, 28, 32]
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
    # making a copy of df4, because may run this cell a couple of times as I develop it
    df5 = df4.copy()
    # Get one hot encoding of columns 'soil_qual_high'
    one_hot1 = pd.get_dummies(df5["soil_qual_high"])
    # Drop original as it is now encoded
    df5 = df5.drop("soil_qual_high", axis=1)
    # Join the encoded df
    df5 = df5.join(one_hot1)
    print(len(df5))
    print(df5.head())
    print(df5.columns.tolist())
    # output was ['state_name', 'county_name', 'lon', 'lat', 'nutr_ret_high',
    #             'soil_qual_low', 'suit_irrig_high_soy', 'AEZ_1', 'AEZ_2', 'AEZ_3',
    #             'AEZ_4', 'AEZ_5', 'AEZ_6', 'AEZ_7', 'AEZ_8', 'AEZ_9',
    #             4, 5, 6, 7, 8, 9, 10]
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
    # making a copy of df5, because may run this cell a couple of times as I develop it
    df6 = df5.copy()
    # Get one hot encoding of columns 'soil_qual_low'
    one_hot2 = pd.get_dummies(df6["soil_qual_low"])
    # Drop original as it is now encoded
    df6 = df6.drop("soil_qual_low", axis=1)
    # Join the encoded df
    df6 = df6.join(one_hot2)
    print(len(df6))
    print(df6.head())
    print(df6.columns.tolist())
    # output was ['state_name', 'county_name', 'lon', 'lat', 'nutr_ret_high',
    #             'suit_irrig_high_soy', 'AEZ_1', 'AEZ_2', 'AEZ_3', 'AEZ_4',
    #             'AEZ_5', 'AEZ_6', 'AEZ_7', 'AEZ_8', 'AEZ_9',
    #             'SQH_1', 'SQH_2', 'SQH_3', 'SQH_4', 'SQH_5', 'SQH_6', 'SQH_7',
    #              4, 5, 6, 7, 8, 9, 10]
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
    # <span style=color:blue>Archiving this df     </span>
    archives_dir = "ML-ARCHIVES--v01/"
    filename = "state_county_lon_lat_soil.csv"
    df6.to_csv(archives_dir + filename, index=False)
    print("wrote file: ", archives_dir + filename)


    #!/usr/bin/env python
    # coding: utf-8
    # ## <span style=color:blue>Fetching weather data from NASA POWER </span>
    #
    # <span style=color:blue>Start by merging year_state_county_yield.csv and state_county_lon_lat.csv into a df.  Will use this below as the backbone for fetching and formating the weather data</span>
    def curr_timestamp():
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        return formatted_datetime


    yscy_file = "year_state_county_yield.csv"
    scll_file = "state_county_lon_lat.csv"
    df_yscy = pd.read_csv(archive_dir + yscy_file)
    df_scll = pd.read_csv(archive_dir + scll_file)
    # Recall that when getting the lon-lat, I changed the name of "DU PAGE, ILLINOIS" to "DUPAGE, ILLINOIS"
    # I will make the same name substitution in df_yscy
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
    # print(df_yscyll.head(30))
    # sanity check - that lon/lat's in new df correspond to the lon/lat's from table state_county_lon_lat.csv
    print(df_yscyll[df_yscyll["year"] == 2022].head(10))
    print(df_scll.head(10))
    # checking on the DU PAGE county entries
    print(df_yscyll.iloc[279:284].head())
    yscyll_filename = "year_state_county_yield_lon_lat.csv"
    df_yscyll.to_csv(archive_dir + yscyll_filename, index=False)
    print("wrote file: ", archive_dir + yscyll_filename)
else:
    pass
yscyll_filename = "year_state_county_yield_lon_lat.csv"
df_yscyll = pd.read_csv(archive_dir + yscyll_filename)

# ### <span style=color:blue>Building a function that, given a year and a lon-lat, pulls 7 momth's worth of weather data from NASA POWER for that lon-lat.  We focus on 2003 to 2022. We pull data only for the growing season for soybeans, which is April to October.
#
# <span style=color:blue>(I am inferring the growing season for my 7 states based on https://crops.extension.iastate.edu/encyclopedia/soybean-planting-date-can-have-significant-impact-yield which is about Iowa, which is about central north-south in my region of interest.)</span>


# setting up a URL template for making requests to NASA POWER

# growing season from April to October

"""
import json

working_dir = 'NASA-POWER/OUTPUTS/'
county_file = 'county_lat_long.csv'

dfcty = pd.read_csv(working_dir + county_file)
# print(dfcty.head())
"""

# see https://gist.github.com/abelcallejo/d68e70f43ffa1c8c9f6b5e93010704b8
#   for available parameters
# I will focus on the following parameters
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

# Now setting up parameterized URLs which will pull weather data,
# focused on growing season, which is April to October
# following
#     https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/
base_url = r"https://power.larc.nasa.gov/api/temporal/daily/point?"
base_url += (
    "parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,GWETROOT,EVPTRNS,ALLSKY_SFC_PAR_TOT&"
)
base_url += "community=RE&longitude={longitude}&latitude={latitude}&start={year}0401&end={year}1031&format=JSON"
# print(base_url)


def fetch_weather_county_year(year, state, county):
    row = df_yscyll.loc[
        (df_yscyll["state_name"] == state)
        & (df_yscyll["county_name"] == county)
        & (df_yscyll["year"] == year)
    ]
    # print(row)
    # print(type(row))
    lon = row.iloc[0]["lon"]
    lat = row.iloc[0]["lat"]
    # print(lon, lat)
    api_request_url = base_url.format(longitude=lon, latitude=lat, year=str(year))

    # this api request returns a json file
    response = requests.get(url=api_request_url, verify=True, timeout=30.00)
    # print(response.status_code)
    content = json.loads(response.content.decode("utf-8"))

    # print('\nType of content object is: ', type(content))
    # print(json.dumps(content, indent=4))

    # print('\nKeys of content dictionary are: \n', content.keys())
    # print('\nKeys of "properties" sub-dictionary are: \n', content['properties'].keys())
    # print('\nKeys of "parameter" sub-dictionary are: \n', content['properties']['parameter'].keys())
    #

    weather = content["properties"]["parameter"]

    df = pd.DataFrame(weather)
    return df


# sanity check
df = fetch_weather_county_year(2022, "ILLINOIS", "LEE")

# examining the output
print(len(df))

print(df.head())


# <span style=color:blue>Create dictionary keyed by the indices of the year_state_county_yield_lon_lat df, and where each entry is a df of weather info pulled from NASA POWER for the given year and lon-lat</span>


# w_df will be a dictionary of df's, each holding weather info for
#    one year-state-county triple
# The dictionary keys will be the index values of df_yscyll that we
#    built above (also archived in year_state_county_yield_lon_lat.csv)
w_df = {}

# will archive each w_df[i] value into a csv as we go along, for 2 reasons:
#   - because this takes forever to run
#   - network connectivity or other errors in middle of run may abort the process

out_dir = archive_dir + "WEATHER-DATA--v01/"
os.makedirs(out_dir, exist_ok=True)
filename = r"weather-data-for-index__{index}.csv"

starttime = datetime.datetime.now().strftime("%Y-%m-% %H:%M:%S")

# for i in range(0,len(df_yscyll)):
# for i in range(278,280):    # had to fix issue of DU PAGE county...
# for i in range(280,len(df_yscyll)):
# for i in range(1779,1780):  # when running big loop it failed at 1779; but worked in this run; network issue?
# for i in range(1779, len(df_yscyll)): # blocked at 2534
# for i in range(2534,2535):    # when running big loop it failed at 1779; but worked in this run; network issue?
for i in range(0, len(df_yscyll)):
    row = df_yscyll.iloc[i]
    outfilename = out_dir + filename.format(index=str(i).zfill(4))

    if Path(outfilename).exists():
        continue
    # print(row)
    w_df[i] = fetch_weather_county_year(
        row["year"], row["state_name"], row["county_name"]
    )
    w_df[i].to_csv(outfilename)
    # adding this to get a feeling of forward progress
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

"""
# sanity check

index = 2
print(r'The contents of yscyll for index {index} is:'.format(index=index), '\n')
print(df_yscyll.iloc[index])

print(r'The head of the weather data for index {index} is:'.format(index=index), '\n')
print(w_df[index].head(10))
"""

endtime = datetime.datetime.now().strftime("%Y-%m-% %H:%M:%S")
print("start and end times were: ", starttime, endtime)


#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook has code for taking the weather data that was archived in part-04, aggregates by MONTHY, and re-formats to form a very wide row for each year-state-county triple.  We finish by creating a wide table for ML, that includes year-state-county, yield, soil data, and weather data </span>

# <span style=color:blue>First step is to pull import the weather files that were created for each year-state-county triple     </span>


yscyll_filename = "year_state_county_yield_lon_lat.csv"

weather_dir = archive_dir + "WEATHER-DATA--v01/"
wdtemplate = r"weather-data-for-index__{padded}.csv"

df_yscyll = pd.read_csv(archive_dir + yscyll_filename)
print(df_yscyll.shape)


w_df = {}
for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    w_df[i] = pd.read_csv(weather_dir + wdtemplate.format(padded=padded))
    # Want to have a name for the index of my dataframe
    w_df[i].rename(columns={"Unnamed: 0": "date"}, inplace=True)
    # w_df[i] = w_df[i].rename_axis(index='DATE')


print(w_df[4].shape)
print(w_df[4].head())


# <span style=color:blue>Test of grouping by MONTH </span>


# takes as input a dataframe whose index field is called "date" and
#   holds 8-character dates, and with columns
#   ['T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'GWETROOT', 'EVPTRNS', 'ALLSKY_SFC_PAR_TOT']
# produces dataframe with same shape, but the values are grouped by MONTH,
#   with a particular aggregation used for each column


def create_monthly_df(df):
    df1 = df.copy()
    # convert index to datetime format
    df1.index = pd.to_datetime(df["date"], format="%Y%m%d")
    # use 'M' for monthly, use 'W' for weekly
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

    # convert index back to string format YYYYMM
    df1_monthly.index = df1_monthly.index.strftime("%Y%m%d")

    return df1_monthly


print(create_monthly_df(w_df[4]).head(50))


# <span style=color:blue>Function that creates a list of all the column names I want for the MONTHLY weather data.    </span>


df_t0 = w_df[0]
cols_narrow = df_t0.columns.values.tolist()[1:]
print(cols_narrow)


df_t1 = create_monthly_df(df_t0)  # dfw['0001']
print(len(df_t1))
# print(df_t1.head())

cols_wide = []
for i in range(0, len(df_t1)):
    row = df_t1.iloc[i]
    # print(row)
    # can't use date, because it has year built in, and weeks start on different numbers...
    month_id = "month_" + str(i).zfill(2)
    # print(date)
    for c in cols_narrow:
        cols_wide.append(month_id + "__" + c)

print(cols_wide)
print(len(cols_wide))


# <span style=color:blue>Function that takes in weather data for one year-state-city triple and produces list of all the MONTHLY weather values, in correct order     </span>


# starts with a df with the weekly aggregates for weather params,
# and produces a long sequence of all the MONTHLY weather values, in order corresponding to cols_wide

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


# sanity check
dfw = create_monthly_df(w_df[0])
print(dfw.head(10))

seqw = create_weather_seq_for_monthly(dfw)
print(json.dumps(seqw, indent=4))


# <span style=color:blue>Building a dictionary that has indexes from df_yscy as keys, and the MONTHLY weather sequences as values    </span>


u_df = {}  # each entry will hold a df corresponding to a weather .csv file
dfw = (
    {}
)  # each entry will hold the df corresponding to monthly aggregation of a weather .csv file
seqw = {}  # each entry will hold the "flattening" of the monthly aggregation df


for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    # print(padded)
    u_df[padded] = pd.read_csv(weather_dir + wdtemplate.format(padded=padded))
    # Want to have a name for the index of my dataframe
    u_df[padded].rename(columns={"Unnamed: 0": "date"}, inplace=True)

    dfw[padded] = create_monthly_df(u_df[padded])
    # print(dfw.head())

    seqw[i] = create_weather_seq_for_monthly(dfw[padded])
    # print(json.dumps(dictw, indent=4)

    # exceeding some I/O threshold
    if i % 30 == 0:
        time.sleep(0.05)

    if i > 9000 and i % 100 == 0:
        time.sleep(0.5)

    if i % 500 == 0:
        print("Completed processing for index: ", i)


# sanity check
# print(json.dumps(seqw, indent=4))


print(len(seqw))
print(json.dumps(seqw, indent=4))


# <span style=color:blue>Converting the dictionary with a sequence for each year-state-county triple into a df     </span>


print(dfw["0000"].shape)
print(len(cols_wide))
print(len(df_yscyll))
print(len(seqw[0]))


df_wide_weather_monthly = pd.DataFrame.from_dict(
    seqw, orient="index", columns=cols_wide
)

print(df_wide_weather_monthly.shape)

print(df_wide_weather_monthly.head())


# <span style=color:blue>Merge projection of state-county-lat-lon-soil table into projection of yscyll table</span>


sclls_file = "state_county_lon_lat_soil.csv"

df_scsoil = pd.read_csv(archive_dir + sclls_file).drop(columns=["lon", "lat"])
print(df_scsoil.shape)
# print(df_scsoil.head())

# will continue working with df_yscyll because updated DU PAGE county
#     (and might update other things in future versions...)

df_ysc_y_soil = pd.merge(
    df_yscyll, df_scsoil, on=["state_name", "county_name"], how="left"
)

df_ysc_y_soil = df_ysc_y_soil.drop(columns=["lon", "lat"])


print(df_ysc_y_soil.shape)
print(df_ysc_y_soil.head())


# <span style=color:blue>Merge df_wide_weather_monthly into df_ysc_y_soil</span>


df_ysc_y_soil_weather_monthly = pd.concat(
    [df_ysc_y_soil, df_wide_weather_monthly], axis="columns"
)

print(df_ysc_y_soil_weather_monthly.shape)
# print(df_ysc_y_soil_weather_monthly.head(10))
print(df_ysc_y_soil_weather_monthly.loc[28:32, :])


# <span style=color:blue>Write the resulting table for ML learning to disk</span>


ml_tables_dir = archive_dir + "ML-TABLES--v01/"
os.makedirs(ml_tables_dir, exist_ok=True)
ml_file = "ML-table-monthly.csv"

df_ysc_y_soil_weather_monthly.to_csv(ml_tables_dir + ml_file, index=False)

print("Wrote file ", ml_tables_dir + ml_file)


#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook has code for taking the weather data that was archived in part-04, aggregates it by WEEK, and re-formats it to form a very wide row for each year-state-county triple.  We finish by creating a wide table for ML, that includes year-state-county, yield, soil data, and weather data </span>


yscyll_filename = "year_state_county_yield_lon_lat.csv"

weather_dir = archive_dir + "WEATHER-DATA--v01/"
wdtemplate = r"weather-data-for-index__{padded}.csv"

df_yscyll = pd.read_csv(archive_dir + yscyll_filename)
print(df_yscyll.shape)


w_df = {}
for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    w_df[i] = pd.read_csv(weather_dir + wdtemplate.format(padded=padded))
    # Want to have a name for the index of my dataframe
    w_df[i].rename(columns={"Unnamed: 0": "date"}, inplace=True)
    # w_df[i] = w_df[i].rename_axis(index='DATE')


print(w_df[4].shape)
print(w_df[4].head())


# <span style=color:blue>Test of grouping by WEEK   </span>


# takes as input a dataframe whose index field is called "date" and
#   holds 8-character dates, and with columns
#   ['T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'GWETROOT', 'EVPTRNS', 'ALLSKY_SFC_PAR_TOT']
# produces dataframe with same shape, but the values are grouped by WEEK,
#   with a particular aggregation used for each column


def create_weekly_df(df):
    df1 = df.copy()
    # convert index to datetime format
    df1.index = pd.to_datetime(df["date"], format="%Y%m%d")
    # use 'M' for monthly, use 'W' for weekly
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

    # convert index back to string format YYYYMM
    df1_weekly.index = df1_weekly.index.strftime("%Y%m%d")

    return df1_weekly


print(create_weekly_df(w_df[4]).head(50))


# <span style=color:blue>Function that creates a list of all the column names I want for the WEEKLY weather data.    </span>


df_t0 = w_df[0]
cols_narrow = df_t0.columns.values.tolist()[1:]
print(cols_narrow)


df_t1 = create_weekly_df(df_t0)  # dfw['0001']
# print(df_t1.head())

cols_wide = []
for i in range(0, len(df_t1)):
    row = df_t1.iloc[i]
    # print(row)
    # can't use date, because it has year built in, and weeks start on different numbers...
    week_id = "week_" + str(i).zfill(2)
    # print(date)
    for c in cols_narrow:
        cols_wide.append(week_id + "__" + c)

print(cols_wide)


# <span style=color:blue>Function that takes in weather data for one year-state-city and produces list of all the WEEKLY weather values, in correct order     </span>


# starts with a df with the weekly aggregates for weather params,
# and produces a long sequence of all the WEEKLY weather values, in order corresponding to cols_wide

print(w_df[0].columns.tolist()[1:])
print(w_df[0].shape)
print(create_weekly_df(w_df[0]).shape)


def create_weather_seq_for_weekly(dfw):
    seq = []
    for i in range(0, len(dfw)):
        for c in cols:
            seq.append(dfw.iloc[i][c])
    return seq


# sanity check
dfw = create_weekly_df(w_df[0])
print(dfw.head(10))

seqw = create_weather_seq_for_weekly(dfw)
print(json.dumps(seqw, indent=4))


# <span style=color:blue>Building a dictionary that has indexes from df_yscy as keys, and the WEEKLY weather sequences as values    </span>


u_df = {}
dfw = {}
seqw = {}


for i in range(0, len(df_yscyll)):
    padded = str(i).zfill(4)
    # print(padded)
    u_df[padded] = pd.read_csv(weather_dir + wdtemplate.format(padded=padded))
    # Want to have a name for the index of my dataframe
    u_df[padded].rename(columns={"Unnamed: 0": "date"}, inplace=True)

    dfw[padded] = create_weekly_df(u_df[padded])
    # print(dfw.head())

    seqw[i] = create_weather_seq(dfw[padded])
    # print(json.dumps(dictw, indent=4)

    # exceeding some I/O threshold
    # if i % 30 == 0:
    #     time.sleep(0.05)

    # if i > 4000 and i % 100 == 0:
    #     time.sleep(0.5)

    if i % 100 == 0:
        print("Completed processing of index ", i)

# sanity check
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


# <span style=color:blue>Where are the NaN values coming from in the above?   </span>
#
# <span style=color:blue>It is because for some years the 7 months end up creating 31 weeks whereas for others only 30 are created.  So we will simply drop week_31.  </span>


# print(cols_wide_weekly)
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


# <span style=color:blue>Merge projection of state-county-lat-lon-soil table into projection of yscyll table</span>


sclls_file = "state_county_lon_lat_soil.csv"

df_scsoil = pd.read_csv(archive_dir + sclls_file).drop(columns=["lon", "lat"])
print(df_scsoil.shape)
# print(df_scsoil.head())

# will continue working with df_yscyll because updated DU PAGE county
#     (and might update other things in future versions...)

df_ysc_y_soil = pd.merge(
    df_yscyll, df_scsoil, on=["state_name", "county_name"], how="left"
)

df_ysc_y_soil = df_ysc_y_soil.drop(columns=["lon", "lat"])


print(df_ysc_y_soil.shape)
print(df_ysc_y_soil.head())


# <span style=color:blue>Merge df_wide_weather_weekly into df_ysc_y_soil</span>


df_ysc_y_soil_weather_weekly = pd.concat(
    [df_ysc_y_soil, df_wide_weather_weekly], axis="columns"
)

print(df_ysc_y_soil_weather_weekly.shape)
# print(df_ysc_y_soil_weather_weekly.head(10))
print(df_ysc_y_soil_weather_weekly.loc[28:32, :])


# <span style=color:blue>Write the resulting table for ML learning to disk</span>


ml_tables_dir = archive_dir + "ML-TABLES--v01/"

ml_file = "ML-table-weekly.csv"

df_ysc_y_soil_weather_weekly.to_csv(ml_tables_dir + ml_file, index=False)

print("Wrote file ", ml_tables_dir + ml_file)


#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook is doing pre-processing of the MONTHLY ML table, including getting all ranges into [0,1].  Then it uses several ML regression algorithms to create predictive models </span>


# pulled this long list from somewhere and added to it.  Not using everything here


# <span style=color:blue>Importing the MONTHLY ML table </span>


ml_tables_dir = archive_dir + "ML-TABLES--v01/"

ml_file = "ML-table-monthly.csv"


df_ml = pd.read_csv(ml_tables_dir + ml_file)
print(df_ml.shape)
print(df_ml.head())

# Happily, there are no null values in my ML table
print(df_ml.isnull().values.any())
# yields False


# <span style=color:blue>Separating my independent variables from my dependent variable     </span>


X = df_ml.drop(columns=["yield"])
y = df_ml.loc[:, ["yield"]]

print(X.shape)
print(y.shape)

# print(X.head())
# print(y.head())


# <span style=color:blue>Separating a training set from a testing set    </span>


# For this pipeline I will do random shuffling of the input records before separating the test set
# Choosing random_state=0 (or any specific integer) will ensure that different runs will use same shuffle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# keeping a copy of y_test, because it may get modified below
y_test_orig = y_test.copy()

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

print(y_test_orig.head())
# note: index of first row in y_test_orig is 7397)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_test.iloc[0, 0])


# <span style=color:blue>"Normalizing" all the scalar values.  Using StandardScalar, which maps the data to have a mean of 0 and standard deviation of 1.  Will use MinMaxScaler below </span>
#
# <span style=color:blue>Note: probably I should have done the scaling before separating the training from the test sets.  One reason I am doing it here is that below I will switch to MinMaxScaler, and make an interesting observation.</span>


X_train = X_train.drop(columns=["year", "state_name", "county_name"])
X_test = X_test.drop(columns=["year", "state_name", "county_name"])


scalerXST = StandardScaler().fit(X_train)
scaleryST = StandardScaler().fit(y_train)

X_trainST = scalerXST.transform(X_train)
y_trainST = scaleryST.transform(y_train)
X_testST = scalerXST.transform(X_test)
y_testST = scaleryST.transform(y_test)


# <span style=color:blue>How to get back to original values of y_test ...    </span>


# testing how inverse of the scaling is working

# basically, if scalery was your scaling function, then use scalery.inverse_transform;
#   NOTE: this works on a sequence

print(df_ml.iloc[7397]["yield"])  #  the first entry in y_test has index 1277 from df_ml
print(y_testST[0])  # .loc[[1277]])
print(scaleryST.inverse_transform(y_testST)[0])


# <span style=color:blue>Trying Lasso...   </span>


# confusingly, you set the "lambda" variable of LASSO algorithm using the parameter "alpha"
# alpha can take values between 0 and 1; using 1.0 is "full penalty", so maximum attempts to remove features
# lassoST = Lasso(alpha=1.0)
# lassoST = Lasso(alpha=0.5)
# lassoST = Lasso(alpha=0.2)
lassoST = Lasso(alpha=0.1)
lassoST.fit(X_trainST, y_trainST)


y_predST = lassoST.predict(X_testST)

print(y_predST)


rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))

print(rmseST)


# from chatGPT!


def plot_predictions(y_test, y_pred, descrip_of_run):
    # Check if the arrays have the same length
    if len(y_test) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    # Create a scatter plot
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="red", linestyle="--")  # Line y_pred = y_test
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Predicted vs Actual for " + descrip_of_run)
    plt.grid(True)
    plt.show()


# Example usage
# y_test = np.array([1, 2, 3, 4, 5])
# y_pred = np.array([1.1, 1.9, 3.2, 3.8, 4.9])

plot_predictions(y_testST, y_predST, "Lasso with StandardScalar")


# <span style=color:blue>Let's try plain old linear regression    </span>


linearST = LinearRegression()


print(type(y_trainST))

linearST.fit(X_trainST, y_trainST)


# <span style=color:blue>As a small side note, when I first created the monthly ML table, I forgot to do one-hot encoding of the 3 soil columns.  Without the one-hot my rmseST was 0.704. So, three soil columns are providing a small boost.   </span>


y_predST = linearST.predict(X_testST)

rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))
print(rmseST)

plot_predictions(y_testST, y_predST, "Linear Regression using StandardScaler")


# ### <span style=color:blue>I now switch to the MinMaxScaler.  A primary reason is so that I can use RRSME.  (Recall that the formula for RRSME is RSME/mean, but with StandardScalar the mean is set to 0 by design.)    </span>


scalerXMM = MinMaxScaler().fit(X_train)
scaleryMM = MinMaxScaler().fit(y_train)

X_trainMM = scalerXMM.transform(X_train)
y_trainMM = scaleryMM.transform(y_train)
X_testMM = scalerXMM.transform(X_test)
y_testMM = scaleryMM.transform(y_test)


# <span style=color:blue>How to get back to original values of y_test when using MinMax ...    </span>


# testing how inverse of the scaling is working with MinMaxScaler

print(df_ml.iloc[7397]["yield"])  #  the first entry in y_test has index 1277 from df_ml
print(y_testMM[0])
print(scaleryMM.inverse_transform(y_testMM)[0])


# <span style=color:blue>Now trying linear regression on MinMaxScaler      </span>
#
# <span style=color:blue>Note: when I ran this without the one-hot soil columns, I get the values 0.0927668152923395, 0.185533630584679, 0.5171931008533303 for rmseMM, rrmseMM and r2MM, respectively <span style=color:red>


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


# <span style=color:blue>Note: interestingly, the shape of the blue blob for linearMM looks identical to the shape of the blue blob for linearST.     </span>

# ### <span style=color:blue>Now trying random forest


# random forest regressor
regrMM = RandomForestRegressor(max_depth=2, random_state=0)
#   with depth 2
#      0.11349253447526553
#      0.22698506895053105
#      0.2773587277559947
# regrMM = RandomForestRegressor(max_depth=10, random_state=0)
#   with depth 10:
#      0.06593145947060583
#      0.13186291894121166
#      0.7561214796081714
# regrMM = RandomForestRegressor(max_depth=20, random_state=0)
#   with depth 20:
#      0.06060039414793919
#      0.12120078829587838
#      0.7939659164434344

# for some reason, need to use y_trainMM.ravel() rather than simply y_trainMM
regrMM.fit(X_trainMM, y_trainMM.ravel())

y_predMM = regrMM.predict(X_testMM)
rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Random Forest Regressor using MinMaxScaler")


# <span style=color:blue>Question: Why is it that with both Linear Regression and Random Forest the slope of the blue blob is more shallow than the slope of the red line?  Is there something we should do about it?     </span>


#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook is doing pre-processing of the WEEKLY ML table, including getting all ranges into [0,1].  Then it uses several ML regression algorithms to create predictive models </span>


# pulled this long list from somewhere and added to it.  Not using everything here


# <span style=color:blue>Importing the WEEKLY ML table </span>


ml_tables_dir = archive_dir + "ML-TABLES--v01/"

ml_file = "ML-table-weekly.csv"


df_ml = pd.read_csv(ml_tables_dir + ml_file)
print(df_ml.shape)
print(df_ml.head())

# Happily, there are no null values in my ML table
print(df_ml.isnull().values.any())
# yields False


# <span style=color:blue>Separating my independent variables from my dependent variable     </span>


X = df_ml.drop(columns=["yield"])
y = df_ml.loc[:, ["yield"]]

print(X.shape)
print(y.shape)

# print(X.head())
# print(y.head())


# <span style=color:blue>Separating a training set from a testing set    </span>


# For this pipeline I will do random shuffling of the input records before separating the test set
# Choosing random_state=0 (or any specific integer) will ensure that different runs will use same shuffle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# keeping a copy of y_test, because it may get modified below
y_test_orig = y_test.copy()

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

print(y_test_orig.head())
# note: index of first row in y_test_orig is 7397)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_test.iloc[0, 0])


# <span style=color:blue>"Normalizing" all the scalar values.  Using StandardScalar, which maps the data to have a mean of 0 and standard deviation of 1.  Will use MinMaxScaler below </span>
#
# <span style=color:blue>Note: probably I should have done the scaling before separating the training from the test sets.  One reason I am doing it here is that below I will switch to MinMaxScaler, and make an interesting observation.</span>


X_train = X_train.drop(columns=["year", "state_name", "county_name"])
X_test = X_test.drop(columns=["year", "state_name", "county_name"])


scalerXST = StandardScaler().fit(X_train)
scaleryST = StandardScaler().fit(y_train)

X_trainST = scalerXST.transform(X_train)
y_trainST = scaleryST.transform(y_train)
X_testST = scalerXST.transform(X_test)
y_testST = scaleryST.transform(y_test)


# <span style=color:blue>How to get back to original values of y_test ...    </span>


# testing how inverse of the scaling is working

# basically, if scalery was your scaling function, then use scalery.inverse_transform;
#   NOTE: this works on a sequence

print(df_ml.iloc[7397]["yield"])  #  the first entry in y_test has index 1277 from df_ml
print(y_testST[0])  # .loc[[1277]])
print(scaleryST.inverse_transform(y_testST)[0])


# <span style=color:blue>Trying Lasso...   </span>


# confusingly, you set the "lambda" variable of LASSO algorithm using the parameter "alpha"
# alpha can take values between 0 and 1; using 1.0 is "full penalty", so maximum attempts to remove features
# lassoST = Lasso(alpha=1.0)
# lassoST = Lasso(alpha=0.5)
# lassoST = Lasso(alpha=0.2)
lassoST = Lasso(alpha=0.1)
lassoST.fit(X_trainST, y_trainST)


y_predST = lassoST.predict(X_testST)

print(y_predST)


rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))

print(rmseST)


# from chatGPT!


def plot_predictions(y_test, y_pred, descrip_of_run):
    # Check if the arrays have the same length
    if len(y_test) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    # Create a scatter plot
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color="red", linestyle="--")  # Line y_pred = y_test
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("Predicted vs Actual for " + descrip_of_run)
    plt.grid(True)
    plt.show()


# Example usage
# y_test = np.array([1, 2, 3, 4, 5])
# y_pred = np.array([1.1, 1.9, 3.2, 3.8, 4.9])

plot_predictions(y_testST, y_predST, "Lasso with StandardScalar")


# <span style=color:blue>Let's try plain old linear regression    </span>


linearST = LinearRegression()


print(type(y_trainST))

linearST.fit(X_trainST, y_trainST)


#


y_predST = linearST.predict(X_testST)

rmseST = math.sqrt(mean_squared_error(y_testST, y_predST))
print(rmseST)

plot_predictions(y_testST, y_predST, "Linear Regression using StandardScaler")


# ### <span style=color:blue>I now switch to the MinMaxScaler.  A primary reason is so that I can use RRSME.  (Recall that the formula for RRSME is RSME/mean, but with StandardScalar the mean is set to 0 by design.)    </span>


scalerXMM = MinMaxScaler().fit(X_train)
scaleryMM = MinMaxScaler().fit(y_train)

X_trainMM = scalerXMM.transform(X_train)
y_trainMM = scaleryMM.transform(y_train)
X_testMM = scalerXMM.transform(X_test)
y_testMM = scaleryMM.transform(y_test)


# <span style=color:blue>How to get back to original values of y_test when using MinMax ...    </span>


# testing how inverse of the scaling is working with MinMaxScaler

print(df_ml.iloc[7397]["yield"])  #  the first entry in y_test has index 1277 from df_ml
print(y_testMM[0])
print(scaleryMM.inverse_transform(y_testMM)[0])


# <span style=color:blue>Now trying linear regression on MinMaxScaler      </span>
#
# <span style=color:blue>Note: when I ran this without the one-hot soil columns, I get the values 0.0927668152923395, 0.185533630584679, 0.5171931008533303 for rmseMM, rrmseMM and r2MM, respectively <span style=color:red>


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


# <span style=color:blue>Note: interestingly, the shape of the blue blob for linearMM looks identical to the shape of the blue blob for linearST.     </span>

# ### <span style=color:blue>Now trying random forest


# random forest regressor
# regrMM = RandomForestRegressor(max_depth=2, random_state=0)
#   with depth 2
#      0.11073969374036899
#      0.22147938748073798
#      0.3119899078797479
# regrMM = RandomForestRegressor(max_depth=10, random_state=0)
#   with depth 10:
#      0.062301254023809476
#      0.12460250804761895
#      0.7822381741106176
regrMM = RandomForestRegressor(max_depth=20, random_state=0)
#   with depth 20:
#      0.06060039414793919
#      0.12120078829587838
#      0.7939659164434344

# for some reason, need to use y_trainMM.ravel() rather than simply y_trainMM
regrMM.fit(X_trainMM, y_trainMM.ravel())

y_predMM = regrMM.predict(X_testMM)
rmseMM = math.sqrt(mean_squared_error(y_testMM, y_predMM))
rrmseMM = rmseMM / (0.5)
r2MM = r2_score(y_testMM, y_predMM)
print(rmseMM)
print(rrmseMM)
print(r2MM)


plot_predictions(y_testMM, y_predMM, "Random Forest Regressor using MinMaxScaler")


# <span style=color:blue>Question: Why is it that with both Linear Regression and Random Forest the slope of the blue blob is more shallow than the slope of the red line?  Is there something we should do about it?     </span>
