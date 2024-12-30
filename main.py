import logging
import os
import pandas as pd

from Caching.Payload import Payload
from Portfolio.BasePortfolio import BasePortfolio

logging.basicConfig(level=logging.DEBUG)

logger = logging.Logger(__name__)


def main():
    selected_columns = ['Ticker', 'Quantity', 'Type', 'Location']
    # Create an object of BasePortfolio for ICLN
    base_portfolio = BasePortfolio(
        "ICLN", "Data/ICLN_Holding_1108.csv", selected_columns, True, us_only_=True, composition_date_="11-08-2024"
    )
    # Cache the securities we get from base Portfolio - can be used later instead of recalculating the security price diff each time.
    cache_base_portfolio_security(base_portfolio)

    base_portfolio_performance = base_portfolio.get_portfolio_performance(percent = 5)
    base_portfolio_performance.calculate_value_at_risk(True)
    base_portfolio_performance.export_performance_to_csv("base")

    run_gradient_ascent2(base_portfolio_performance)
    run_gradient_ascent(10000, base_portfolio_performance)

def run_gradient_ascent2(base_portfolio_performance):
    list_of_security = base_portfolio_performance.composition_df_.index.values
    columns = list(list_of_security) + ["VaR"]
    file_path = "Data/Gradient2.csv"

    # Initialize the CSV file with headers if it doesn't exist
    if not os.path.isfile(file_path):
        df_header = pd.DataFrame(columns=columns)
        df_header.to_csv(file_path, index=False)

    point_str = base_portfolio_performance.gradient_ascent_2()
    var_value = Payload.get_var_cache()[point_str]
    # Prepare the data for this iteration
    weights = [float(w) for w in point_str.split(',')]
    weights.append(var_value)  # Add the VaR value
    # Create a DataFrame for the current iteration
    df_row = pd.DataFrame([weights], columns=columns)
    # Append the row to the CSV file without writing the header
    df_row.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Gradient 2 written to CSV.")

def run_gradient_ascent(n, base_portfolio_performance):
    list_of_security = base_portfolio_performance.composition_df_.index.values
    columns = list(list_of_security) + ["VaR"]
    file_path = "Data/Gradient1.csv"

    # Initialize the CSV file with headers if it doesn't exist
    if not os.path.isfile(file_path):
        df_header = pd.DataFrame(columns=columns)
        df_header.to_csv(file_path, index=False)

    for i in range(n):
        point_str = base_portfolio_performance.gradient_ascent(plot_graph=True, simulation_n=i)
        var_value = Payload.get_var_cache()[point_str]

        # Prepare the data for this iteration
        weights = [float(w) for w in point_str.split(',')]
        weights.append(var_value)  # Add the VaR value

        # Create a DataFrame for the current iteration
        df_row = pd.DataFrame([weights], columns=columns)

        # Append the row to the CSV file without writing the header
        df_row.to_csv(file_path, mode='a', header=False, index=False)

        # Optional: Print progress
        print(f"Iteration {i + 1}/{n} written to CSV.")


def cache_base_portfolio_security(base_portfolio: BasePortfolio):
    """
    This function will iterate over all assets in base_portfolio and
    First run on Payload.get_security_cache() will ensure all assets have been loaded and cached
    """
    asset_codes = list(base_portfolio.holdings_df_.index)  # list of assets in the portfolio

    for asset_code in asset_codes:
        Payload.get_security_cache(ticker=asset_code)


if __name__ == "__main__":
    main()
