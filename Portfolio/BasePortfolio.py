import pandas as pd
import logging
from Caching.Payload import Payload
from Caching.PortfolioPerformance import PortfolioPerformance

logging.basicConfig(level=logging.DEBUG)


class BasePortfolio:
    """
    This class will be used when we want to create an instance of managed portfolio with multiple holdings.
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.logger = logging.getLogger(cls.__name__)
        return instance

    def __init__(self, ticker_: str, file_location_: str, selected_columns_: list,
                 equity_only_: bool, us_only_: bool, composition_date_):
        self.ticker_ = ticker_
        self.file_location_ = file_location_
        self.equity_only_ = equity_only_
        self.selected_columns_ = selected_columns_
        self.us_only_ = us_only_
        self.holdings_df_ = self.load_file_to_data_frame()
        self.convert_quantity_to_int()
        self.composition_date_ = pd.to_datetime(composition_date_)
        self.logger.info(f"dataframe - \n {self.holdings_df_}")

    def load_file_to_data_frame(self):
        dataframe = pd.read_csv(self.file_location_, usecols=self.selected_columns_)
        dataframe.set_index('Ticker', inplace=True)
        if self.equity_only_:
            dataframe = dataframe[dataframe['Type'] == 'EQUITY']
        if self.us_only_:
            dataframe = dataframe[dataframe['Location'] == 'United States']

        return dataframe

    def calculate_weight_by_qty(self):
        quantity_sum = self.holdings_df_['Quantity'].sum()
        self.holdings_df_['Weight'] = self.holdings_df_['Quantity'] / quantity_sum

    def convert_quantity_to_int(self):
        """ formats string to integer
        some values includes comma. For example 1,241,252 will be converted to int(1241252)
        may raise exception if there were unsuccessful conversion
        """
        self.holdings_df_['Quantity'] = self.holdings_df_['Quantity'].str.replace(',', '')
        self.holdings_df_['Quantity'] = pd.to_numeric(self.holdings_df_['Quantity'], errors='coerce')
        if self.holdings_df_['Quantity'].isnull().any():
            raise ValueError("Non-numeric values found in the 'Quantity' column")

    def get_portfolio_performance(self, percent) -> PortfolioPerformance:
        """
        Returns portfolio weights
        :return: PortfolioPerformance object
        """
        market_value_ = self.get_fund_market_value()

        performance_df = self.holdings_df_.copy()
        calculated_weights = []

        for index, row in self.holdings_df_.iterrows():
            asset_code = str(index)
            security_info = Payload.get_security_cache(asset_code)
            if not security_info:
                self.logger.error(f"{asset_code} is missing in security cache")
                continue

            df = security_info.nasdaq_security_df
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            close_price = self.get_composition_date_close_price(df)
            weight = row['Quantity'] * close_price / market_value_
            calculated_weights.append(weight)

        performance_df['Weight'] = calculated_weights
        return PortfolioPerformance(self.ticker_, performance_df, base=True, percent=percent)

    def get_composition_date_close_price(self, df: pd.DataFrame) -> float:
        """ Returns closing price of the asset on composition date of the BasePortfolio """
        return df.loc[df['Date'] == self.composition_date_, 'Close/Last'].values[0]

    def get_fund_market_value(self) -> float:
        """
        We use the function to calculate the marketValue the price of the share in the date of the
        fund composition. This marketValue can we used to get the weight
        """
        market_value = 0
        for index, row in self.holdings_df_.iterrows():
            asset_code = str(index)
            security_info = Payload.get_security_cache(asset_code)
            if not security_info:
                self.logger.error(f"{asset_code} is missing in security cache")
                continue

            df = security_info.nasdaq_security_df
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            qty = row['Quantity']
            close_price = self.get_composition_date_close_price(df)
            market_value += qty * close_price
            self.logger.debug(f"calculated {market_value=}, {asset_code=}, {qty=}, {close_price=}")
        return market_value
