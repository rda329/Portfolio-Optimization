from typing import Union

from Caching.NasdaqAsset import NasdaqAsset
from Portfolio.SecurityInfo import SecurityInfo
import logging

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Payload:
    __security_cache = {}
    __portfolio_composition = {}
    __var_cache = {}

    @classmethod
    def set_var_cache(cls, point_string : str, value_at_risk):
        cls.__var_cache[point_string] = value_at_risk

    @classmethod
    def get_var_cache(cls):
        return cls.__var_cache

    @classmethod
    def set_security_cache(cls, security_info : SecurityInfo, ticker : str):
        cls.__security_cache[ticker] = security_info

    @classmethod
    def get_security_cache(cls, ticker: str) -> Union[SecurityInfo, None]:
        """
        Returns SecurityInfo object, `ticker` arg should be valid string.
        If any exception is raised, then it returns `None`.

        Method looks in `__security_cache` dict for security info. If it exists returns it directly.
        If security info doesn't exist, then it loads it from a file (call to func load_asset_historical_data())
        and saves it into `__security_cache` dict
        """
        if not ticker:
             return None

        security_info = cls.__security_cache.get(ticker)
        if security_info:
            return security_info

        asset = NasdaqAsset(ticker, "Data/Nasdaq")
        try:
            df = asset.load_asset_historical_data(force_save_file=False, _format="pandas")
            new_security_info = SecurityInfo(name=ticker, df=df)
            cls.set_security_cache(new_security_info, ticker)
            return new_security_info
        except Exception as e:
            logger.error(repr(e))
