#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook creates a function that, given a county, will generate some number (e.g., 1000) of lon-lat pairs that are all within that county.   </span>

# <span style=color:blue>First, a function that builds an approximate bounding box around a county. </span>
#
# <span style=color:blue>This is a little sloppy - we build a box that is 1 degree x 1 degree that is centered on the central lon-lat of the county.  Most of the counties in my 7-state soy region have this characteristic. </span>

import datetime
import json
import os
import random
import subprocess
from osgeo import gdal, osr
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from pyproj import Transformer
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from shapely.geometry import Point

# from utils import plot_image


def plot_image(image, factor=1, clip_range=(0, 1)):
    # Apply factor to image
    image = image * factor

    # Clip image values
    image = image.clip(clip_range[0], clip_range[1])

    # Plot the image
    plt.imshow(image)
    plt.show()


# will fetch the lon-lats at center of each county from the file state_county_lon_lats.csv

archive_dir = "../ML-pipeline-for-soybean-yield-forecast/ML-ARCHIVES--v01/"
scll = "state_county_lon_lat.csv"

df_scll = pd.read_csv(archive_dir + scll)
print(df_scll.head())


# Geocoding function to retrieve coordinates for a county
def approx_county_bbox(state, county):
    rows = df_scll.loc[
        (df_scll["state_name"] == state) & (df_scll["county_name"] == county)
    ]
    # print(rows)
    lon = rows["lon"].values[0]
    lat = rows["lat"].values[0]
    # print(lon,lat)

    if True:
        west_lon = lon - 0.5
        east_lon = lon + 0.5
        north_lat = lat + 0.5
        south_lat = lat - 0.5
        return {
            "center_lon": lon,
            "center_lat": lat,
            "west_lon": west_lon,
            "east_lon": east_lon,
            "north_lat": north_lat,
            "south_lat": south_lat,
        }
    else:
        print("no lat-lon found for ", state, county)
        return {"error": "no lat-lon found for " + county + ", " + state}


# test for Bureau County, IL
# center point lon for this county is: -89.5341179
# center point lat for this county is:  41.4016294

bbox = approx_county_bbox("ILLINOIS", "JO DAVIESS")
# bbox = approx_county_bbox('ILLINOIS', 'FAKE NAME')

print(json.dumps(bbox, indent=4, sort_keys=True))
# ### <span style=color:blue>Now working towards a function that tests if lat-lon is in a county    </span>
#
# <span style=color:blue>As a first step, I downloaded files from https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html that hold polygon specifications for all of the US counties.  In particular, I fetched the Counties file that was 1:20,000,000 at the link https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip.  (This was the least precise, and don't see a need for more precision.) From inside the zip directory I retrieved, the ".dbf" file seemed most useful. </span>
# downloaded this from
county_dir = "COUNTY-BOUNDING-POLYGONS/"
os.makedirs(county_dir, exist_ok=True)
county_file = "cb_2022_us_county_20m.dbf"
county_path = county_dir + county_file

# Load county boundary data from Shapefile
counties = gpd.read_file(county_path)

# Print column names
print(counties.head())
# <span style=color:blue>The state_name, county_name values from the USDA NASS yield data are all capitals, and need to convert to the format above, which is first-letter-is-capitalized     </span>

# test
print("NEW JERSEY".title())
print("DU PAGE".title())
# <span style=color:blue>Function to test with a given lon-lat is in a state-county     </span>

# Load county boundary data; this is a .dbf file

# downloaded this from
county_dir = "COUNTY-BOUNDING-POLYGONS/"
county_file = "cb_2022_us_county_20m.dbf"
county_path = county_dir + county_file
counties = gpd.read_file(county_path)


def lon_lat_in_county(longitude, latitude, state_name, county_name):
    # Load county boundary data; this is a .dbf file
    counties = gpd.read_file(county_path)

    # Find the specified county
    county = counties[
        (counties["NAME"] == county_name.title())
        & (counties["STATE_NAME"] == state_name.title())
    ]
    # print(county)

    if county.empty:
        print(f"County '{county_name}' not found.")
        return False

    # Create shapely point from the provided latitude and longitude
    point = Point(longitude, latitude)

    # Check if the point is within the county polygon
    return point.within(county.geometry.values[0])


# test
state_name = "ILLINOIS"
county_name = "JO DAVIESS"
lon_in = -90.174374
lat_in = 42.350666
lon_out = -95
lat_out = 35

print(lon_lat_in_county(lon_in, lat_in, state_name, county_name))
print(lon_lat_in_county(lon_out, lat_out, state_name, county_name))
# <span style=color:blue>Function that generates some number of lon-lat pairs that are within a county     </span>


# assumes state_name, county_name are all-caps, as in the USDA NASS yield data sets
def gen_lon_lat_in_county(state_name, county_name, count):
    list = []
    bbox = approx_county_bbox(state_name, county_name)
    # print(json.dumps(bbox, indent=4, sort_keys=True))
    for i in range(0, count):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        # print(r1,r2)
        lon = round(bbox["east_lon"] + r1 * (bbox["west_lon"] - bbox["east_lon"]), 7)
        lat = round(bbox["south_lat"] + r2 * (bbox["north_lat"] - bbox["south_lat"]), 7)
        list += [[lon, lat]]
    return list


# test
list = gen_lon_lat_in_county("ILLINOIS", "JO DAVIESS", 1000)
print(json.dumps(list[0:5], indent=4))
print()
print(json.dumps(list[995:1000], indent=4))
# <span style=color:blue>     </span>

# <span style=color:blue>Based on the file state_county_lon_lat.csv, build a dictionary with shape state / county / seq_of_lon_lat_in_county.  Actually, this cell is a warm up.    </span>

print(df_scll.state_name.unique())
# answer is: ['ILLINOIS' 'INDIANA' 'IOWA' 'MISSOURI' 'NEBRASKA' 'OHIO']

# oh - realizing now that somehow Minnesota got dropped from my set of states
# It was in my notebook ML-for-soybeans-part-01--fetching-yield-data, where
# I mispelled MINNESTOTA.  Not fixing it for now...

dict = {}
for state in df_scll.state_name.unique():
    dict[state] = {}

print(json.dumps(dict, indent=4, sort_keys=True))
# <span style=color:blue>Here is a function that walks through all the state-county pairs of df_scll, and for each one creates a sequence of 1000 lon-lats in that state-county, and puts that into dict.     </span>


def create_lon_lat_seqs(count):
    dict = {}
    for state in df_scll.state_name.unique():
        dict[state] = {}
    for i in range(0, len(df_scll)):
        row = df_scll.iloc[i]
        # print(row)
        state = row["state_name"]
        county = row["county_name"]
        dict[state][county] = gen_lon_lat_in_county(state, county, count)
        if i % 50 == 0:
            print(f"Have completed generation of {str(i)} sequences of lon-lats")
    return dict


print(datetime.datetime.now())
dict = create_lon_lat_seqs(5000)
print(datetime.datetime.now())

# print(json.dumps(dict, indent=4, sort_keys=True))
# <span style=color:blue>Save dict as json  </span>

out_file = "state_county__seq_of_lon_lats.json"

with open(archive_dir + out_file, "w") as fp:
    json.dump(dict, fp)


#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>This notebook builds the function soy_is_here(year,lat,lon), which produces "True" if the 100m x 100m area around lon-lat was a soybean field in the given year.  (At least, according to the data of USDA NASS.)  </span>
#
# ## <span style=color:blue>Then we import the dictionary with the lon-lat sequences for each county, and for each year find the first 20 that are in soybean fields, and right

# <span style=color:blue>First step is to create a function that tests, given a year-lon-lat triple, whether there was a soy field at lat-lon during the given year.  This is based on checking files downloaded from https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php.  To understand the meaning of the pixel values, please see https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/meta.php and the files in there, e.g., https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_ia22.htm.  Among other things, you will see that the value of '5' corresponds to soybean fields </span>

# first, examining the structure of the files I downloaded

dir_main = "CROPSCAPE/DATA-DOWNLOADS/"
os.makedirs(dir_main, exist_ok=True)


# following the structure of the directory name and file from downloaded zip files, which are organized by year
def pathname_for_year(year):
    last_dir_name = f"{str(year)}_30m_cdls/"
    file_name = f"{str(year)}_30m_cdls.tif"
    return dir_main + last_dir_name + file_name


# test
print(pathname_for_year(2022))
# <span style=color:blue>Now we inspect structure of the tif files.     </span>
#
# #### <span style=color:blue>Note that the Coordinate Reference System (CRS) is EPSG: 5070 rather than EPSG:4326 (also basically equivalent to WGS84), which is the one we are often using.  BTW, the unit of measure for EPSG:5070 is 1 meter, and so the pixels in these tif files are approximately 30m x 30m.  (See https://epsg.io/5070-1252) </span>


def pull_useful_gdal(dataset):
    useful = {}

    # Get raster band count
    useful["band_count"] = dataset.RasterCount

    # Get size
    useful["size"] = [dataset.RasterXSize, dataset.RasterYSize]

    # Get corner coordinates
    geotransform = dataset.GetGeoTransform()
    useful["proj:transform"] = geotransform
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    corners = {
        "upperLeft": [geotransform[0], geotransform[3]],
        "lowerLeft": [geotransform[0], geotransform[3] + geotransform[5] * y_size],
        "upperRight": [geotransform[0] + geotransform[1] * x_size, geotransform[3]],
        "lowerRight": [
            geotransform[0] + geotransform[1] * x_size,
            geotransform[3] + geotransform[5] * y_size,
        ],
    }

    # Compute center coordinates
    corners["center"] = [
        (corners["upperLeft"][0] + corners["lowerRight"][0]) / 2,
        (corners["upperLeft"][1] + corners["lowerRight"][1]) / 2,
    ]
    useful["cornerCoordinates"] = corners

    # Define the source and target spatial references
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(dataset.GetProjection())
    print("Dataset project", dataset.GetProjection())

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326) #4326  # WGS84 #  , 8826

    # Set up a coordinate transformation
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)

    # Get corner coordinates and reproject them
    corners_projected = {}
    for name, coords in [('upperLeft', (0, 0)), ('lowerLeft', (0, dataset.RasterYSize)),
                         ('upperRight', (dataset.RasterXSize, 0)), ('lowerRight', (dataset.RasterXSize, dataset.RasterYSize))]:
        x, y = gdal.ApplyGeoTransform(geotransform, *coords)
        lon, lat, _ = transform.TransformPoint(x, y)
        corners_projected[name] = [lon, lat]
    useful["cornerCoordinatesProj"] = corners_projected

    # Get bounding box
    # bbox = {
    #     'west_longitude': min(corners_projected['upperLeft'][0], corners_projected['lowerLeft'][0], corners_projected['upperRight'][0], corners_projected['lowerRight'][0]),
    #     'north_latitude': max(corners_projected['upperLeft'][1], corners_projected['lowerLeft'][1], corners_projected['upperRight'][1], corners_projected['lowerRight'][1]),
    #     'east_longitude': max(corners_projected['upperLeft'][0], corners_projected['lowerLeft'][0], corners_projected['upperRight'][0], corners_projected['lowerRight'][0]),
    #     'south_latitude': min(corners_projected['upperLeft'][1], corners_projected['lowerLeft'][1], corners_projected['upperRight'][1], corners_projected['lowerRight'][1])
    # }
    bbox = {
        'west_longitude': corners_projected['upperLeft'][0],
        'north_latitude': corners_projected['upperLeft'][1],
        'east_longitude': corners_projected['lowerRight'][0],
        'south_latitude': corners_projected['lowerRight'][1]
    }
    useful["bbox"] = bbox
    bbox = {
        'west_longitude': corners['upperLeft'][0],
        'north_latitude': corners['upperLeft'][1],
        'east_longitude': corners['lowerRight'][0],
        'south_latitude': corners['lowerRight'][1]
    }
    useful["bbox_noproj"] = bbox

    # Get EPSG code
    useful["espgEncoding"] = int(src_srs.GetAuthorityCode(None))

    return useful

def pull_useful(
    ginfo,
):  # should give as input the result.stdout from calling gdalinfo -json
    useful = {}
    useful["band_count"] = len(ginfo["bands"])
    useful["cornerCoordinates"] = ginfo["cornerCoordinates"]
    useful["proj:transform"] = ginfo["stac"]["proj:transform"]
    useful["size"] = ginfo["size"]
    useful["bbox"] = ginfo["stac"]["proj:projjson"]["bbox"]
    useful["espgEncoding"] = ginfo["stac"]["proj:epsg"]
    return useful


path_to_file = pathname_for_year(2008)
# path_to_file = pathname_for_year(2022)
dataset = gdal.Open(path_to_file)
useful_gdal = pull_useful_gdal(dataset)
gdalInfoReq = " ".join(["gdalinfo", "-json", path_to_file])
# print(gdalInfoReq[k])
result = subprocess.run([gdalInfoReq], shell=True, capture_output=True, text=True)
# print(result)
print()
# print(result.stdout)
# print(result.stdout)
print(result.stderr)
gdalInfo = json.loads(result.stdout)

useful = pull_useful(gdalInfo)
with open("gdal_process.json", "w") as outfile:
    json.dump(useful, outfile, indent=2, sort_keys=True)
with open("gdal_lib.json", "w") as outfile:
    json.dump(useful_gdal, outfile, indent=2, sort_keys=True)
# exit()
# print(json.dumps(useful, indent=2, sort_keys=True))
# print(json.dumps(useful_gdal, indent=2, sort_keys=True))

# <span style=color:blue>Function to transform from EPSG:4326 to EPSG:5070.  The rasterio-based function we use below will take coordinates in EPSG:5010, since the tif files we are using here are in EPSG:5010.     </span>

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070")


def from_4326_to_5070(lon, lat):
    # I'm not sure why the role positions of lon-lat are different on input and output
    # but that is what my numerous small test runs showed to me
    new_lon, new_lat = transformer.transform(lat, lon)
    return new_lon, new_lat


# test on coordinates from central Iowa
old_lon = -92.8
old_lat = 42.7
print(from_4326_to_5070(old_lon, old_lat))
# (you can check this at https://epsg.io/transform)
# <span style=color:blue>Function that fetches a 3x3 square of pixel values from the given tif file.  The pixels in the tif file correspond to  30m x 30m, so we are looking at a rouhgly 100m x 100m area that is all or mostly soybean field </span>
#
# <span style=color:blue>Note that in 2008 the target area was planted mainly with maize, but in 2022 it was planted with soybeans</span>


# expects lon-lat to be in EPSG:4326.
# These are converted to EPSG:5070 inside the function
def get_coordinate_pixels(tiff_file, lon, lat):
    dataset = rasterio.open(tiff_file)
    lon_new, lat_new = from_4326_to_5070(lon, lat)
    # print(lon_new,lat_new)
    py, px = dataset.index(lon_new, lat_new)
    # print(py, px)
    # create 3px x 3px window centered on the lon-lat
    window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
    clip = dataset.read(window=window)
    return clip


# test
old_lon = -92.8
old_lat = 42.7
path_to_file = pathname_for_year(2008)
print(get_coordinate_pixels(path_to_file, old_lon, old_lat))
print()
path_to_file = pathname_for_year(2022)
print(get_coordinate_pixels(path_to_file, old_lon, old_lat))
# <span style=color:blue>Also, a function that that tests whether all 9 spots in the 3x3 square have a given value.  (We are interested in "5", which is soy beans.)</span>


# land_use_val should be an integer; see, e.g.,
#     https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_ia22.htm
#     for mapping from values to meanings
def usage_is_here(year, lon, lat, land_use_val):
    path_to_file = pathname_for_year(year)
    arr = get_coordinate_pixels(path_to_file, lon, lat)
    out = True
    for i in range(0, 3):
        for j in range(0, 3):
            out = out & (arr[0][i][j] == land_use_val)
    return out


def soy_is_here(year, lon, lat):
    return usage_is_here(year, lon, lat, 5)


old_lon = -92.8
old_lat = 42.7
print(soy_is_here(2008, old_lon, old_lat))
print(soy_is_here(2022, old_lon, old_lat))
# ### <span style=color:blue>Importing the dictionary with lon-lat sequences.  Also setting a second dict that will hold lists lon-lats that are in soybean fields.</span>

in_file = "state_county__seq_of_lon_lats.json"

f = open(archive_dir + in_file)
dict = json.load(f)

print(dict.keys())
# <span style=color:blue>Function that scans through one list of lon-lats and finds first set that are in soybean fields</span>


def gen_soy_lon_lats(year, state, county, count):
    list = dict[state][county]
    i = 0
    out_list = []
    for ll in list:
        if soy_is_here(year, ll[0], ll[1]):
            out_list += [ll]
            i += 1
        if i == 20:
            return out_list, []
    print(
        f"\nFor {str(year)}, {state}, {county}: \nFailed to find {str(count)} lon-lats that were in soybean fields. Found only {str(i)}.\n"
    )
    short_fall_record = [year, state, county, i]
    return out_list, short_fall_record


list, short = gen_soy_lon_lats(2008, "ILLINOIS", "MASSAC", 20)
print(list)
print(short)
print()
list, short = gen_soy_lon_lats(2008, "MISSOURI", "DALLAS", 20)
print(list)
print(short)
# <span style=color:blue>Function that generates a fixed number of lon-lats in soybean fields for each year and each county. This took quite a while to run completely -- about 4 hours.    </span>

working_dir = "OUTPUTS/OUTPUT-v01/"
os.makedirs(working_dir, exist_ok=True)
dict1_file = "year_state_county_soy_seq.json"
short_list = "year_state_county_shortfalls.json"


def gen_all_soy_lists(dict, count):
    dict1 = {}
    for year in range(2008, 2023):
        dict1[year] = {}
        for key in dict.keys():
            dict1[year][key] = {}
    print(dict1.keys())
    print(dict1[2013].keys())

    shortfall_list = []

    i = 0
    for year in dict1.keys():
        for state in dict.keys():
            for county in dict[state].keys():
                list, short = gen_soy_lon_lats(year, state, county, count)
                dict1[year][state][county] = list
                if short != []:
                    shortfall_list += [short]

                i += 1
                if i % 20 == 0:
                    print(
                        f"Have generated soybean lon-lat lists for {str(i)} year-county pairs"
                    )
                if i % 50 == 0:
                    with open(working_dir + dict1_file, "w") as fp:
                        json.dump(dict1, fp)
                    with open(working_dir + short_list, "w") as fp:
                        json.dump(shortfall_list, fp)

    return dict1, shortfall_list


print(datetime.datetime.now())
dict1, short = gen_all_soy_lists(dict, 20)
print(datetime.datetime.now())
# <span style=color:blue>Save the dict1 and also the shortfalls    </span>

dict1_file = "year_state_county_soy_seq.json"
short_list = "year_state_county_shortfalls.json"

with open(archive_dir + dict1_file, "w") as fp:
    json.dump(dict1, fp)
with open(archive_dir + short_list, "w") as fp:
    json.dump(short, fp)
# <span style=color:blue>Collecting year-state-county with zero hits </span>

zero_falls = []

for l in short:
    if l[3] == 0:
        zero_falls += [[l]]

print(len(zero_falls))

print(json.dumps(zero_falls, indent=4))

zero_file = "year_state_county_soy_zero_falls.json"
with open(archive_dir + zero_file, "w") as fp:
    json.dump(zero_falls, fp)
# <span style=color:blue>Checking if any year-state-county in zero_falls had a positive yield in year_state_county_yield table</span>

yscy_file = "year_state_county_yield.csv"

df_yscy = pd.read_csv(archive_dir + yscy_file)
print("Top of df_yscy")
print(df_yscy.head())

zero_with_yield = []
for l in zero_falls:
    year = l[0][0]
    state = l[0][1]
    county = l[0][2]
    rows = df_yscy[
        (df_yscy["year"] == year)
        & (df_yscy["state_name"] == state)
        & (df_yscy["county_name"] == county)
    ]
    if len(rows) > 0:
        y = rows["yield"].iloc[0]
        zero_with_yield += [
            {"year": year, "state_name": state, "county_name": county, "yield": y}
        ]

print("\nLength of zero_with_yield is: ", len(zero_with_yield))
print("\nListing of zero_with_yield")
df_zwy = pd.DataFrame(zero_with_yield)
print(df_zwy.head(30))

zero_with_yield = "year_state_county_soy_zero_with_yield.csv"
df_zwy.to_csv(archive_dir + zero_with_yield, index=False)
# <span style=color:blue>For this exercise, we will drop these year-state-county triples from consideration.  A more thorough approach would be to focus on these year-state-county pairs (and perhaps the other ones with < 20 lon-lats), and randomly generate more lon-lats within the county until at least a few are found inside soybean fields.  (On the one hand, there have to be some if there was a yield ... however, CropScape is not perfect and may not have identified them accurately.)</span>

#!/usr/bin/env python
# coding: utf-8

# ## <span style=color:blue>In this notebook, we illustrate how to get the NDVI value for a single cell of size roughly 100m x 100m.  This will give you the basic machinery needed to gather sequences of NDVI values that can be incorporated into your ML pipelines   </span>

# ### <span style=color:blue>First, we create function that retrieves the NVDI for a given year, week, and 100m x 100m cell centered at some lon-lat   </span>

# <span style=color:blue>To get started with accessing SentinelHub using Python, I found the site https://sentinelhub-py.readthedocs.io/en/latest/index.html to be helpful.  In particular, you can find and download the SentinelHub-py github repository at https://github.com/sentinel-hub/sentinelhub-py, and then work through some of the Examples.</span>
#
# <span style=color:blue>First, we set up access to SentinelHub</span>

# To access SentinelHub you need a client_id and client_secret.
# To get your own access to SentinelHub, go
#    to https://docs.sentinel-hub.com/api/latest/api/overview/authentication/.
#    From there you can get a user name and password for a free 30-day trial.
#    Once you sign in, find you way to
#    https://apps.sentinel-hub.com/dashboard/, and from the
#    "User Settings" area you can create an OAuth client -- this will give you
#    a Client_ID and a Client_Secret.  (I had to create a couple of these in order
#    to get one that worked.  Also, they do expire after a while...)
# I put my client_ID and client_secret into some environment variables

SENTINEL_CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
SENTINEL_CLIENT_SECRET = os.environ.get("SENTINEL_CLIENT_SECRET")
# <span style=color:blue>Now create a client for accessing SentinelHub     </span>

config = SHConfig()

# using third client id and secret, from 2023-05-26
config.sh_client_id = SENTINEL_CLIENT_ID
config.sh_client_secret = SENTINEL_CLIENT_SECRET

if not config.sh_client_id or not config.sh_client_secret:
    print(
        "Warning! To use Process API, please provide the credentials (OAuth client ID and client secret)."
    )
else:
    print("Successfully set up SentinelHub client")
# <span style=color:blue> Not sure why sentinelhub examples include this next cell, but I will blindly imitate them   </span>


# <span style=color:blue>Importing useful things from SentinelHub  </span>

# The following is not a package. It is a file utils.py which should be in the same
#     folder as this notebook.
# As a slight variation, I have cloned the sentinelhub-py repo into my local github,
#     and grab utils.py from there
# ### <span style=color:blue>As a small warm-up exercise, to help you get familiar with accessing data from SentinalHub, here is an example of pulling some RGB data and viewing it     </span>
#
# <span style=color:blue> First, identifying a couple of bounding boxes to work with. Note that the second example is focusing on a single cell with size about 100m x 100m. </span>

# convenient site for finding lat/long coordinates:
#     http://bboxfinder.com/#0.000000,0.000000,0.000000,0.000000

# Bounding box containing the 7 soybean states of interest is as follows:
# [-104.370117,35.782171,-79.628906,48.048710]
# However, this is too big of a region to request in one call to SentinelHub

# Bounding box for about 1/8 of Iowa
# [-96.481934,42.520700,-95.075684,43.516689]

corner_iowa_coords_wgs84 = (-96.481934, 42.520700, -95.075684, 43.516689)

# a central point in IOWA
# [-96.350098,42.195969,-93.801270,43.484812]

# by using corners that are .001 apart, with 100 meter resultion, I get a box with
#    size 1 x 1 pixels
#    size 8 x 11 pixels

# The following lon-lat is in Buena Vista county, Iowa,
# and was a soybean field in 2022 (but apparently not in 2021...)

lon = -94.7386486
lat = 42.6846289

# building a .001 degree x .001 degree bbox around that
point_iowa_wgs84 = (-94.738, 42.684, -94.737, 42.685)
# central_iowa_wgs84 = (-94.73,42.68,--94.74,42.69)
# <span style=color:blue>Once you have the bounds for a box, then you can initialize a "BBox" object and specify both its bounds and also the desired resolution.    </span>
#
# <span style=color:blue>Note: by experimenting I found that you cannot request an image where the box has > 2500 pixels along either direction.  Also, each pixel can be at most 200 m x 200 m.  This puts an effective limit on the size of box you can retrieve with one call to SentinelHub -- about 500km x 500km at the equator.   </span>

# using 150m, because when I used 200m the actual request used 203m,
#     which exceeded the bound on pixel size
resolution1 = 150
corner_iowa_bbox = BBox(bbox=corner_iowa_coords_wgs84, crs=CRS.WGS84)
corner_iowa_size = bbox_to_dimensions(corner_iowa_bbox, resolution=resolution1)

resolution2 = 100
point_iowa_bbox = BBox(bbox=point_iowa_wgs84, crs=CRS.WGS84)
point_iowa_size = bbox_to_dimensions(point_iowa_bbox, resolution=resolution2)
print(
    f"For corner Iowa box, image shape at {resolution1} m resolution: {corner_iowa_size} pixels"
)
print()

print(
    f"For point Iowa box, image shape at {resolution2} m resolution: {point_iowa_size} pixels"
)
# <span style=color:blue>Getting RGB for corner_iowa_bbox     </span>
#
# <span style=color:blue>Here is some helpful text from the example notebook "process_request.ipynb" that I have been following for this part of my notebook </span>
#
# We build the request according to the API Reference, using the SentinelHubRequest class. Each Process API request also needs an evalscript.
#
# The information that we specify in the SentinelHubRequest object is:
#
#     an evalscript,
#     a list of input data collections with time interval,
#     a format of the response,
#     a bounding box and it's size (size or resolution).
#
# The evalscript in the example is used to select the appropriate bands. We return the RGB (B04, B03, B02) Sentinel-2 L1C bands.
#
# With request_true_color_1_day, the image from Jun 12th 2020 is downloaded. Without any additional parameters in the evalscript, the downloaded data will correspond to reflectance values in UINT8 format (values in 0-255 range).
#
# <span style=color:blue>I am also experimenting with request_true_color_7_day, to see what happens if my interval is multiple days. </span>

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color_1_day = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2020-06-12", "2020-06-13"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=corner_iowa_bbox,
    size=corner_iowa_size,
    config=config,
)

request_true_color_7_day = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2020-06-12", "2020-06-19"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=corner_iowa_bbox,
    size=corner_iowa_size,
    config=config,
)
# <span style=color:blue>Invoking these two functions    </span>

corner_iowa_color_imgs_1_day = request_true_color_1_day.get_data()
corner_iowa_color_imgs_7_day = request_true_color_7_day.get_data()
# <span style=color:blue>Exploring the outputs.  It appears that the multi-day gives back the sums of the values for the days that are contributing.    </span>

print(
    f"Returned data is of type = {type(corner_iowa_color_imgs_1_day)} and length {len(corner_iowa_color_imgs_1_day)}."
)
print(
    f"Single element in the list is of type {type(corner_iowa_color_imgs_1_day[-1])} and has shape {corner_iowa_color_imgs_1_day[-1].shape}"
)
print()
print(corner_iowa_color_imgs_1_day)


print(
    f"Returned data is of type = {type(corner_iowa_color_imgs_7_day)} and length {len(corner_iowa_color_imgs_7_day)}."
)
print(
    f"Single element in the list is of type {type(corner_iowa_color_imgs_7_day[-1])} and has shape {corner_iowa_color_imgs_7_day[-1].shape}"
)
print()
print(corner_iowa_color_imgs_7_day)
# <span style=color:blue>To plot first single-day image, we have to  get the values to be between 0 and 1.  In fact, we first scale to (0,1) but then multiply by 3.5 to brighten the picture    </span>

ci_image_1_day = corner_iowa_color_imgs_1_day[0]
print(f"Type of each value in ci_image_1_day: {ci_image.dtype}")

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(ci_image_1_day, factor=3.5 / 255, clip_range=(0, 1))
# <span style=color:blue>Let's look at how the 7-day interval turns out...   </span>

ci_image_7_day = corner_iowa_color_imgs_7_day[0]
print(f"Type of each value in ci_image_7_day: {ci_image.dtype}")

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(ci_image_7_day, factor=1 / 255, clip_range=(0, 1))
# ### <span style=color:blue>Oh my - the clouds are dominating the top part of image!! Please see the example of building cloud masks in the file process_request.ipynb in the Examples area of the sentinelhub-py github repo.    </span>

# <span style=color:blue>Building a sentinel request to pull NDVI values for a single 100m x 100m cell.  Recall the formula for NDVI is (B08 - B04) / (B08 + B04).</span>

# Recall that we built point_iowa_bbox (along with point_iowa_size)
# to be a single pixel of size 100m x 100m, that was in a soybean field in 2022

# It is centered at
lon = -94.7386486
lat = 42.6846289

evalscript_NVDI_bands = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04", "B08"]
            }],
            output: {
                bands: 2
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B08];
    }
"""

request_NVDI_corner = SentinelHubRequest(
    evalscript=evalscript_NVDI_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2020-06-12", "2020-06-14"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=corner_iowa_bbox,
    size=corner_iowa_size,
    config=config,
)

request_NVDI_point = SentinelHubRequest(
    evalscript=evalscript_NVDI_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2022-08-02", "2022-08-03"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=point_iowa_bbox,
    size=point_iowa_size,
    config=config,
)

print(request_NVDI_point)
data = request_NVDI_point.get_data()
print(data)
print()
B04 = data[0][0][0][0]
B08 = data[0][0][0][1]
print(B04, B08)
NVDI = (B08 - B04) / (B08 + B04)
print("\nNVDI is: ", NVDI)

print()

# print(request_NVDI_corner)
# print(request_NVDI_corner.get_data())
# ### <span style=color:blue>Note: if you run the above request on point_iowa_box for "2022-08-01" to "2022-08-02", then you get [0,0].  I think this is because the satellite didn't go over this cell on that one day. Remember that with the two sentinel-2 satellites taken together there is a 5-day return rate. In general, one should probably make single-cell requests that are across a 5 day span, e.g., 2022-04-01 to 2022-04-06. </span>
