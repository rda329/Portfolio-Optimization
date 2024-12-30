import os
import csv
import logging
import urllib
from datetime import datetime
from io import StringIO
from typing import Literal

import pandas
import requests
logging.basicConfig(level=logging.DEBUG)


class NasdaqAsset:
    """
    class NasdaqAsset provides helpers to download asset historical data from Nasdaq, saves under

    usage:
        FSLR = NasdaqAsset("FSLR")
        csv_con = FSLR.load_asset_historical_data(force_save_file=False, _format="pandas")
        # force_save_file=True   => download and save file even if exists
        # force_save_file=False  => use existing file if exists, (downloads and saves if not)
        # _format="json"         => returns list of dictionary
        # _format="pandas"       => returns pandas dataframe
        # _format="csv"          => returns csv string

        print(csv_con[:100])
    """
    logger: logging.Logger
    data_path: str
    _default_data_path: str = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + "/Data/Nasdaq/"
    asset_code: str

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.logger = logging.getLogger(cls.__name__)
        return instance

    def __init__(self, asset_code: str, data_path: str = None) -> None:
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self._default_data_path
            self.logger.warning(f"No data path provided defaulting to {self._default_data_path}")

        self.asset_code = asset_code

        self.check_data_path()  # ensure nasdaq asset directory exists

    def check_data_path(self):
        """ checks data_path, if path does not exist then creates a new directory"""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            self.logger.info(f"Created data path {self.data_path}")

    def load_asset_historical_data(
            self, force_save_file: bool = False, _format: Literal["json", "pandas", "csv"] = "pandas"
    ) -> list | str | pandas.DataFrame:
        """
        loads asset's historical data. from file if exists, if not downloads file and saves it for later use (caching)
        returns csv string or list of parsed json
        """
        asset_file_path = os.path.join(self.data_path, f"{self.asset_code}.csv")
        asset_file_exists = os.path.exists(asset_file_path)
        if not asset_file_exists or force_save_file:
            json_data = self.request_download_data()
            csv_content = self.convert_to_csv(json_data)
            self.write_to_file(asset_file_path, csv_content)
        else:
            # load from file
            with open(asset_file_path, "r") as f:
                csv_content = f.read()

        if _format == "json":
            return self.csv_to_json(csv_content)
        elif _format == "pandas":
            return pandas.read_csv(StringIO(csv_content))

        return csv_content

    def compose_nasdaq_data_download_url(self):
        """ prepares nasdaq api url that downloads historical data of the asset """
        date_today = datetime.today().date()
        to_date = date_today.isoformat()
        from_date = date_today.replace(year=date_today.year - 10).isoformat()
        base_url = f"https://api.nasdaq.com/api/quote/{self.asset_code}/historical"
        query_params = {
            "assetclass": "stocks",
            "fromdate": from_date,
            "limit": "9999",
            "todate": to_date,
        }
        url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
        return url

    def request_download_data(self):
        url = self.compose_nasdaq_data_download_url()
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if not response.ok:
            raise requests.HTTPError(f"Could not download Nasdaq asset {self.asset_code} data: {response.text=}")

        response_json = response.json()
        return response_json

    def write_to_file(self, file_path: str, content: str) -> None:
        with open(file_path, "w") as f:
            f.write(content)
            self.logger.info(f"Saved Nasdaq asset {self.asset_code} data to {file_path}")

    def convert_to_csv(self, nasdaq_json: dict) -> str:
        self.logger.debug(f"Converting Nasdaq asset {self.asset_code} data to CSV")
        data: dict = nasdaq_json.get("data", {})
        if not data:
            raise ValueError(f"Nasdaq asset {self.asset_code} data is empty {nasdaq_json}")
        trading_table: dict = data.get("tradesTable")
        headers: dict = trading_table.get("headers")
        rows: list = trading_table.get("rows")

        # map row keys to header names
        rows = [{headers[key]: self.clean_value(value) for key, value in row.items()} for row in rows]

        fieldnames = list(headers.values())
        with StringIO() as csv_buffer:
            dict_writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(rows)
            csv_buffer.seek(0)
            csv_content_buff = csv_buffer.read()

        return csv_content_buff

    @staticmethod
    def clean_value(value: str) -> str:
        value = value.strip().replace("$", "").replace(",", "")
        return value

    def csv_to_json(self, csv_content: str) -> list:
        self.logger.debug(f"Converting csv content {self.asset_code} data to JSON")
        with StringIO() as csv_buffer:
            csv_buffer.write(csv_content)
            csv_buffer.seek(0)
            dict_reader = csv.DictReader(csv_buffer)
            json_content = [row for row in dict_reader]

        return json_content
