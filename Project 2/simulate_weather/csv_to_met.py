import pandas as pd
from jinja2 import Template
import io
import os
import numpy as np
from datetime import datetime as datetime

outputdir = "simulated"

met_file_j2_template = '''[weather.met.weather]
!Date period from: {{ year_from }} to {{ year_to }}
Latitude={{ lat }}
Longitude={{ lon }}
tav={{ tav }}
amp={{ amp }}
year day radn maxt mint rain
() () (MJ^m2) (oC) (oC) (mm)
{{ vardata }}
'''

files = os.listdir("simulated")
files = [x for x in files if ".csv" in x]
for f in files:
    # get lat/lon for later
    lat = f[:5]
    lon = f[6:11]

    # read df
    met_dataframe = pd.read_csv("simulated/" + f)

    # create template
    j2_template = Template(met_file_j2_template)

    # save data to a buffer
    df_output_buffer = io.StringIO()
    met_dataframe.to_csv(df_output_buffer, sep=" ", header=False, na_rep="NaN", index=False, mode='w', float_format='%.1f')

    # Get values from buffer
    # Go back to position 0 to read from buffer
    # Replace get rid of carriage return or it will add an extra new line between lines
    df_output_buffer.seek(0)
    met_df_text_output = df_output_buffer.getvalue()
    met_df_text_output = met_df_text_output.replace("\r\n", "\n")

    # create date column
    met_dataframe["date"] = (np.asarray(met_dataframe['year'], dtype='datetime64[Y]')-1970)+(np.asarray(met_dataframe['day'], dtype='timedelta64[D]')-1)

    # creat month column
    met_dataframe.loc[:, 'month'] = met_dataframe.date.dt.month
    month = met_dataframe.loc[:, 'month']

    # calculate some stuff
    met_dataframe.loc[:, 'tmean'] = met_dataframe[['maxt', 'mint']].mean(axis=1)
    tmeanbymonth = met_dataframe.groupby(month)[["tmean"]].mean()
    maxmaxtbymonth = tmeanbymonth['tmean'].max()
    minmaxtbymonth = tmeanbymonth['tmean'].min()
    amp = np.round((maxmaxtbymonth-minmaxtbymonth), decimals=5)

    # calculate tav
    tav = tmeanbymonth.mean().tmean.round(decimals=5)

    # configure header variables
    current_date = datetime.now().strftime("%d%m%Y")
    year_from = met_dataframe.year.min()
    year_to = met_dataframe.year.max()

    # delete df to free up memory
    del met_dataframe

    in_memory_met = j2_template.render(
                                    lat=lat,
                                    lon=lon,
                                    tav=tav,
                                    amp=amp,
                                    current_date=current_date,
                                    year_from=year_from,
                                    year_to=year_to,
                                    vardata=met_df_text_output
                                    )
    df_output_buffer.close()

    full_output_path = "simulated/" + f[:-4] + ".met"
    with open(full_output_path, 'w+') as f:
        f.write(in_memory_met)