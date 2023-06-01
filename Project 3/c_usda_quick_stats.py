from pathlib import Path
import urllib
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import requests


class c_usda_quick_stats:
    def __init__(self, api_key):
        # Set the USDA QuickStats API key, API base URL, and output file path where CSV files will be written.

        # self.api_key = 'PASTE_YOUR_API_KEY_HERE'
        self.api_key = api_key

        self.base_url_api_get = (
            "http://quickstats.nass.usda.gov/api/api_GET/?key=" + self.api_key + "&"
        )

    def get_data(self, parameters, file_path: Path):
        # Call the api_GET api with the specified parameters.
        # Write the CSV data to the specified output file.

        # Create the full URL and retrieve the data from the Quick Stats server.

        full_url = self.base_url_api_get + parameters
        print(full_url)

        try:
            s_result = urllib.request.urlopen(full_url)

            print(s_result.status, s_result.reason)

            s_text = s_result.read().decode("utf-8")

            # Create the output file and write the CSV data records to the file.

            with file_path.open("w", encoding="utf8") as f:
                f.write(s_text)
        except HTTPError as error:
            print(error.code, error.reason)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching the data: {e}")
        except ValueError as e:
            print(f"Failed to parse the response data: {e}")
        except:
            print(
                "Failed because of unknown exception; perhaps the USDA NASS site is down"
            )