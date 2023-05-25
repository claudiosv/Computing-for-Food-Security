import pandas as pd
import numpy as np
import os
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def flipped_sigmoid(x):
    return 1 / (1 + math.exp(x))

mu=0.0
def gaussian_noise(x, mu, std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy

def simulate(df, row):
    # make a copy of the reference year for the current year
    current_year = df.copy()

    # find the average rain of the reference year
    reference_rain = np.sum(current_year.rain)
    # find the difference between the reference year and the expected avg in GAEZ
    daily_diff_rain = (reference_rain - row.precipitation_mm)/len(current_year.rain)
    # adjust reference year to meet GAEZ expectation
    current_year.rain = current_year.rain - daily_diff_rain

    # fewer rain days -> more random noise
    std = (np.max(current_year.rain) - np.min(current_year.rain))*flipped_sigmoid(row.rain_days_zscore)
    # add noise
    current_year.rain = gaussian_noise(current_year.rain, mu, std)
    # get rid of negative rain
    current_year.loc[current_year.rain < 0, "rain"] = 0

    # find average daily temp
    reference_temp = np.mean((current_year.maxt + current_year.mint)/2)
    # get differene between reference temp and expected temp
    daily_diff_temp = (reference_temp - row.mean_temp)/len(current_year.maxt)

    # adjust reference year to meet GAEZ expectation
    current_year.maxt = current_year.maxt - daily_diff_temp
    current_year.mint = current_year.mint - daily_diff_temp

    # more days of 35+ -> more noise
    std = (np.max(current_year.maxt) - np.min(current_year.maxt))* sigmoid(row.tmax35_zscore)
    current_year.maxt = gaussian_noise(current_year.maxt, mu, std)

    # more days of 15- -> more noise
    std = (np.max(current_year.mint) - np.min(current_year.mint))* sigmoid(row.tmin15_zscore)
    current_year.mint = gaussian_noise(current_year.mint, mu, std)

    # more days of 35+ and fewer days of 15- -> more sun
    avg_sun = (row.tmax35_zscore - row.tmin15_zscore)/2
    std = (np.max(current_year.radn) - np.min(current_year.radn))* sigmoid(avg_sun)
    current_year.radn = gaussian_noise(current_year.radn, mu, std)

    # add column for year
    current_year.loc[:, "year"] = row.year

    return current_year

# break up GAEZ data by RCP
gaez = pd.read_csv("hfdl-esm2m-3.71_-79.61.csv", index_col=0)
gaez_26 = gaez[gaez.rcp == "rcp2p6"]
gaez_85 = gaez[gaez.rcp == "rcp8p5"]

# calculate z scores for each column in GAEZ data
for col in ["precipitation_mm", "rain_days", "mean_temp", "tmax35", "tmin15"]:
    col_zscore = col + '_zscore'
    gaez_26.loc[:, col_zscore] = (gaez_26[col] - gaez_26[col].mean())/gaez_26[col].std(ddof=0)
    gaez_85.loc[:, col_zscore] = (gaez_85[col] - gaez_85[col].mean())/gaez_85[col].std(ddof=0)

# get list of weather files
files = os.listdir("csv_1981-2020")

# loop through files
for filename in files:

    # read in file to pandas df
    df = pd.read_csv("csv_1981-2020/" + filename)
    # get lat and lon for naming the file
    lon = str(df.lon[0])
    lat = str(df.lat[0])

    # filter for 5 year period
    df = df[df.year >= 2015]

    # get relevant columns
    df = df[["day", "radn", "maxt", "mint", "rain"]]

    # get daily average of 5 years to make "reference year"
    df = df.groupby("day")[["radn", "maxt", "mint", "rain"]].mean().reset_index()

    # df to keep track of incremental results
    final_df = pd.DataFrame()

    # simulate each year of RCP 8.5
    for index, row in gaez_85.iterrows():

        # use GAEZ to adjust reference year
        current_year = simulate(df, row)

        # add newly calculated year to running list
        final_df = pd.concat([final_df, current_year], axis=0)

    # save df to csv
    final_df = final_df[["year", "day", "radn", "maxt", "mint", "rain"]]
    final_df.to_csv("simulated/" + lat + "-" + lon + "-predictions_85.csv", index=False)

    # df to keep track of incremental results
    final_df = pd.DataFrame()

    # simulate each year of RCP 2.6
    for index, row in gaez_26.iterrows():

        # use GAEZ to adjust reference year
        current_year = simulate(df, row)

        # add newly calculated year to running list
        final_df = pd.concat([final_df, current_year], axis=0)

    # save df to csv
    final_df = final_df[["year", "day", "radn", "maxt", "mint", "rain"]]
    final_df.to_csv("simulated/" + lat + "-" + lon + "-predictions_26.csv", index=False)
