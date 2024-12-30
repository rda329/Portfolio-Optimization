import logging
from datetime import timedelta
import numpy as np
import pandas
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


class SecurityInfo:
    name: str
    nasdaq_security_df: pandas.DataFrame = None
    _synthetic_data_start_date = pd.Timestamp('2021-01-01')

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.logger = logging.getLogger(cls.__name__)
        return instance

    def __str__(self):
        return f"SecurityInfo({self.name}) rows: {self.nasdaq_security_df.shape[0]} columns: {self.nasdaq_security_df.shape[1]}"

    def __init__(self, name: str, file_location_: str = None, df: pd.DataFrame = None):
        """
        One of following args required

        :param file_location_: (string) provide filepath of the csv file
        :param df: (Dataframe) or provide pandas Dataframe object
        """
        self.name = name
        if file_location_:
            self.nasdaq_security_df = self.load_file_to_data_frame(file_location_)
        elif df is not None:
            self.nasdaq_security_df = df
        else:
            raise ValueError('Please provide either a file path or a data frame')
        self.generate_synthetic_data()  # generate synthetic data for non-existing dates
        self.crop_data()
        self.get_price_diff()

    @staticmethod
    def load_file_to_data_frame(file_location_: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file_location_)
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe.set_index('Date', inplace=True)
        return dataframe

    def get_price_diff(self):
        self.nasdaq_security_df = self.nasdaq_security_df.copy()
        self.nasdaq_security_df.sort_index(inplace=True)
        self.nasdaq_security_df['Price Diff'] = self.nasdaq_security_df['Close/Last'].diff(-1)

    def generate_synthetic_data(self):
        df = self.nasdaq_security_df
        last_record = df.tail(1)
        last_record_date = last_record.iloc[0]["Date"]
        last_record_date = pd.to_datetime(last_record_date)
        if last_record_date <= self._synthetic_data_start_date:
            # skip if data already exists
            return
        self.logger.info(
            f"Generating synthetic data {self.name}. from {self._synthetic_data_start_date.date()} to {last_record_date.date()}")
        close_mean = df.loc[:, 'Close/Last'].mean()
        close_std = df.loc[:, 'Close/Last'].std()
        prev_date = last_record_date  # initial while loop condition
        while prev_date > self._synthetic_data_start_date:
            prev_date = prev_date - timedelta(days=1)
            random_price = self.get_random_security_close_value(close_mean, close_std)
            while prev_date.date().isoweekday() > 5:
                # Monday is 1 and Sunday is 7 => skip if it is weekend days
                # https://docs.python.org/3/library/datetime.html#datetime.datetime.isoweekday
                # keep reducing date by 1 day until it is business days
                prev_date = prev_date - timedelta(days=1)
            formatted_date_str = prev_date.date().strftime("%m/%d/%Y")
            new_df = pd.DataFrame([[formatted_date_str, random_price]], columns=['Date', 'Close/Last'])
            df = pd.concat([df, new_df], ignore_index=True)
        self.nasdaq_security_df = df
        self.logger.info(f"Done generating synthetic data {self.name}")

    @staticmethod
    def get_random_security_close_value(mean: np.float64, std: np.float64) -> np.float64:
        """ using the formula mean - 1.5*std , mean   """
        random_val = np.random.uniform(mean - 1.5 * std, mean)
        random_val = round(random_val, 2)
        return np.float64(random_val)

    def crop_data(self):
        """ remove data prior to  self._synthetic_data_start_date """
        df = self.nasdaq_security_df
        df['date_fmt'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df[~(df['date_fmt'] < self._synthetic_data_start_date)]
        df.drop('date_fmt', axis=1)
        self.nasdaq_security_df = df
