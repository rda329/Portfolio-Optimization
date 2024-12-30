import time
from typing import Union, List, Tuple

from Caching.BoundedList import BoundedList
from Caching.Payload import Payload
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class PortfolioPerformance:

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.logger = logging.getLogger(cls.__name__)
        return instance

    def __init__(self, ticker: str, composition_df_, percent, base: bool):
        if not isinstance(composition_df_, pd.DataFrame):
            raise TypeError("composition_df_ must be a pandas DataFrame")
        self.ticker = ticker
        self.base = base
        self.percent = percent
        self.value_at_risk = None
        self.composition_df_ = composition_df_
        self.portfolio_performance_df = self.calculate_fund_performance_overtime()

    def calculate_fund_performance_overtime(self) -> Union[pd.DataFrame, None]:
        security_values = []
        for index, row in self.composition_df_.iterrows():
            curr_security = Payload.get_security_cache(str(index))

            quantity = row['Quantity']
            security_df = curr_security.nasdaq_security_df.copy()
            # Ensure 'Date' is in datetime format
            security_df['Date'] = pd.to_datetime(security_df['Date'])
            # Sort by date
            security_df.sort_values('Date', inplace=True)
            # Calculate the market value for the security over time
            security_df['MarketValue'] = security_df['Close/Last'] * quantity
            # Set 'Date' as the index
            security_df.set_index('Date', inplace=True)
            # Keep only the 'MarketValue' column
            security_df = security_df[['MarketValue']]
            # Rename the 'MarketValue' column to the security ticker for identification
            security_df.rename(columns={'MarketValue': index}, inplace=True)
            # Append the DataFrame to the list
            security_values.append(security_df)
        if not security_values:
            self.logger.error("No security data available to calculate portfolio performance.")
            return

        # Combine all securities' market values into a single DataFrame aligned by 'Date'
        portfolio_df = pd.concat(security_values, axis=1)

        # drop rows with holiday dates | artificial data cannot be checked for holidays
        # so dropna() will ensure there is no invalid trading days in market
        portfolio_df.dropna(axis=0, inplace=True)

        # Sum across securities to get the total portfolio value per date
        portfolio_df['TotalMarketValue'] = portfolio_df.sum(axis=1)
        portfolio_df['MarketValueDiff'] = portfolio_df['TotalMarketValue'].shift(-1) - portfolio_df['TotalMarketValue']
        return portfolio_df

    def calculate_value_at_risk(self, print_chart=False):
        data = self.portfolio_performance_df['MarketValueDiff'].dropna()
        # Calculate the lower 5% to get value at risk
        value_at_risk = np.percentile(data, self.percent)
        self.value_at_risk = value_at_risk

        if print_chart:
            self.save_histogram_chart(data, value_at_risk)

    def return_portfolio_performance_for_composition(self, weights, filename_addon):
        curr_portfolio_performance = self.create_new_composition(weights)
        curr_portfolio_performance.export_performance_to_csv(filename_addon)

    def save_histogram_chart(self, data, value_at_risk):
        # Create the histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=100)

        # Plot the red line to mark the lower 5%
        plt.axvline(value_at_risk, color='red', label=f'ValueAtRisk: {value_at_risk:.4f}')
        plt.legend()

        # Set the title and labels
        plt.title("Market Value Difference Histogram")
        plt.xlabel("Market Value Difference")
        plt.ylabel("Frequency")

        # Save the histogram to a file
        save_filename = "Data/market_value_difference_histogram.png"
        plt.savefig(save_filename)
        self.logger.info(f"Histogram saved to '{save_filename}'")

    def export_performance_to_csv(self, filename_addon):
        self.portfolio_performance_df.to_csv(f"Data/portfolio_performance_{filename_addon}.csv")

    def create_new_composition(self, new_weight):
        """
            Here we use copy of current PortfolioPerformance object to create a new PortfolioPerformance object.
            We do this by changing the weight with the weight in the parameter.
            Here we also recalculate the qty by multiplying the new weight value by (base qty / base weight).
            This allows us to maintain the qty based on the weight.
        """
        copy_composition_df_ = self.composition_df_.copy()
        copy_composition_df_['new_weight'] = new_weight
        copy_composition_df_['Quantity'] = (
                copy_composition_df_['new_weight'] *
                (copy_composition_df_['Quantity'] / copy_composition_df_['Weight'])
        )
        copy_composition_df_['Weight'] = copy_composition_df_['new_weight']
        copy_composition_df_.drop(columns=['new_weight'], inplace=True)
        return PortfolioPerformance(self.ticker, copy_composition_df_, base=False, percent = self.percent)

    def get_number_of_security(self) -> int:
        return self.composition_df_.shape[0]

    def return_value_at_risk_for_composition(self, weights) -> float:
        """
            This function takes point which is a list of weights and creates a new PortfolioPerformance class from it
            which is used to get value at risk
            We also give a percent parameter to use as the percentile for value at risk
            We also have a cache and if the composition of weights is already in the cache, we just return the value at risk from cache
        """
        point_string = ",".join(map(str, weights))
        var_cache = Payload.get_var_cache()
        if point_string in var_cache:
            return var_cache[point_string]

        curr_portfolio_performance = self.create_new_composition(weights)
        curr_portfolio_performance.calculate_value_at_risk()
        var_cache[point_string] = curr_portfolio_performance.value_at_risk
        return curr_portfolio_performance.value_at_risk

    def get_equal_points(self, size):
        equal_value = round(1 / size, 2)
        points = [equal_value] * size
        return points

    def get_random_points(self, size) -> List[float]:
        random_nums = np.random.rand(size)
        random_nums /= random_nums.sum()  # Normalize to sum to 1

        # Scale to hundredths
        scaled_nums = np.floor(random_nums * 100)

        # Ensure no value is zero after scaling
        scaled_nums[scaled_nums == 0] = 1  # Set any zero value to 1 (smallest possible weight)

        # Adjust the discrepancy
        discrepancy = int(100 - scaled_nums.sum())
        for i in range(discrepancy):
            scaled_nums[i % size] += 1

        # Convert back to decimal values
        final_nums = scaled_nums / 100.0
        return list(final_nums)

    def gradient_ascent(self, plot_graph: bool = False, simulation_n: int = None):
        """
            Here we will create a random composition of the securities
            First we will get the number of securities we have from base portfolio
            Then we will use that to get a random composition which total amounts to 1
            Then we will use that as parameter for return_value_at_risk_for_composition
        """
        n_stocks = self.get_number_of_security()
        point = self.get_random_points(n_stocks)

        # step 1 find var for the current point
        point_var, point_string = self.get_point_var(point)

        self.logger.info(f"Starting Point: {point} \nVaR: {point_var}\n\n")
        stop_condition = False
        epoch = 0
        tolerance = 1e-6
        weight_adjustment_rate = 0.01
        l_abs_diffs_plot = []
        weights_plot = []

        bounded_list = BoundedList(max_size=5)  # for checking last 5 changes in VaR diffs
        bounded_list.items = list(range(5))  # init random 5 different values, so code doesnt stop on first epoch
        while not stop_condition:
            # step 2 : find VaR for all possible combinations
            inter_var_diff = self.calculate_inter_point_vars(n_stocks, point, point_var, weight_adjustment_rate)

            # step 3 : Find the largest absolute difference
            largest_diff = inter_var_diff[0][0]
            location = 0
            diff = inter_var_diff[0]
            # self.logger.debug(inter_var_diff, "diff list")
            for i in range(len(inter_var_diff)):
                check = max(inter_var_diff[i], key=abs)
                if abs(check) > abs(largest_diff):
                    largest_diff = abs(check)
                    diff = check
                    location = i

            # step 4 : find add / subtract locations
            add_loc = location
            sub_loc = inter_var_diff[location].index(diff)

            # step 5 : calculate new point
            if point[sub_loc] - weight_adjustment_rate >= 0 and diff != 0:
                diff_sign = np.sign(diff)  # either 1 or -1
                # if diff > 0 then sub_loc should be subtracted, add_loc should be added
                # otherwise, vice versa
                point[sub_loc] += (-1 * diff_sign * weight_adjustment_rate)
                point[add_loc] += (diff_sign * weight_adjustment_rate)

            # Round all weights to 2 decimal places
            point = self.norm_point_weights(point)

            # update value at risk for the new composition
            point_var, point_string = self.get_point_var(point)

            l_abs_diffs_plot.append(diff)
            weights_plot.append(point)

            # add diff to bounded_list. for checking last 5 changes in VaR diff
            bounded_list.add(diff)

            if abs(diff) < tolerance:
                stop_condition = True
                self.logger.debug(f"epoch: {epoch}\nChange in VaR has converged to {abs(diff)}, approximately 0. "
                                  f"The local min is \n{point}\nThe VaR is {point_var}")
            elif epoch >= 100:
                stop_condition = True
                # Update epoch limit set to 100, this number was set empirically
                self.logger.info(f"epoch limit was met, VaR did not converge")
            elif bounded_list.unique_length == 1:
                # if bounded_list.unique_length returns 1, then it means last 5 epochs nothing has changed in diff
                stop_condition = True
                self.logger.debug(f"epoch: {epoch}\nVaR did not change in last 5 epochs {abs(diff)=}")

            if epoch % 5 == 0:
                # prints every 5 steps
                self.logger.debug(f"epoch : {epoch}, Change : {diff}\nVaR: {point_var}\n point : {point}\n")

            epoch += 1

        if plot_graph and len(l_abs_diffs_plot) > 1:
            if not simulation_n:
                self.logger.error("simulation_n should be provided to use plot_graph")
            else:
                self.plot_weights(weights_plot, simulation_n)
                self.plot_largest_abs_diffs(l_abs_diffs_plot, simulation_n)

        return point_string

    def plot_largest_abs_diffs(self, diffs, simulation_n: int):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=list(range(0, len(diffs))), y=diffs, linewidth=2)

        plt.title('Line Graph of VaR value difference During Gradient Ascent')
        plt.xlabel('Epoch')
        plt.ylabel('VaR difference')
        plt.grid()
        plt.tight_layout()

        save_filename = f"Data/var_diffs_change_line_plot_{simulation_n}.png"
        plt.savefig(save_filename)

    def plot_weights(self, plot_weights, simulation_n: int):
        tickers = list(self.composition_df_.index)

        # Flatten the data and create a list of dictionaries
        data = []
        for i in range(len(plot_weights)):
            row = {
                ticker: plot_weights[i][j]
                for j, ticker in enumerate(tickers)
            }
            row["Epoch"] = i
            data.append(row)

        df = pd.DataFrame(data)
        df_long = df.melt(id_vars='Epoch', var_name='Ticker', value_name='Weight')

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_long, x='Epoch', y='Weight', hue='Ticker', markers=True, palette='hls', markersize=7, linewidth=2,
            alpha=0.8
        )

        plt.title('Line Graph of Weights for Different Tickers During Gradient Ascent')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()

        save_filename = f"Data/weight_changes_line_plot_{simulation_n}.png"
        plt.savefig(save_filename)

    def get_point_var(self, point: List[float]) -> Tuple[float, str]:
        """ if point_string (string key of `point` arg) exists in cache return, otherwise calculate new var and cache"""
        # start_t = time.time()
        var_cache = Payload.get_var_cache()
        point_string = ",".join(map(str, point))
        if point_string in var_cache:
            point_var = var_cache[point_string]
            # print(f"get_point_var() run time (cached): {time.time() - start_t} seconds")
            return point_var, point_string

        # if does not exist in cache, save it and return
        point_var = self.return_value_at_risk_for_composition(point)
        Payload.set_var_cache(point_string, value_at_risk=point_var)
        # print(f"get_point_var() run time: {time.time() - start_t} seconds")
        return point_var, point_string

    def calculate_inter_point_vars(
            self, n_stocks: int, point: List[float], point_var: float, weight_adj_rate: float = 0.01
    ):
        inter_var_diff = []
        # start_t = time.time()
        for i in range(n_stocks):
            inter_point_var_diff_list = []
            for j in range(n_stocks):
                if i == j:
                    continue
                inter_point = point.copy()
                inter_point[i] += weight_adj_rate
                inter_point[j] -= weight_adj_rate

                inter_point = self.norm_point_weights(inter_point)
                # Creating a new PortfolioComposition here to then calculate value at risk
                inter_point_var, inter_point_string = self.get_point_var(inter_point)

                inter_point_var_diff_list.append(inter_point_var - point_var)
            # first list index is location to add step,  second list index is location to subtract
            inter_var_diff.append(inter_point_var_diff_list)

        # print(f"calculate_inter_point_vars() run time: {time.time() - start_t} seconds")
        return inter_var_diff

    @staticmethod
    def norm_point_weights(point: List[float]) -> List[float]:
        """
        Ensure weights are within [0, 1] with precision 2 decimal points
        :param point: List[float] that needs to be within range [0, 1] with precision 2 decimal points
        :return: List[float] converted
        """
        return [max(0, min(1, round(w, 2))) for w in point]

    # This grad descent implementation keeps the total number of shares the same as the starting number, and reallocates to minimize loss, shorting is not allowed

    # Conduct test many times pick top 3 portfolios plus base
    # When comparing results conduct a t-test to see if average return of one portfolio make up is statistically larger than another. (compared to base)
    # Return =  average 1 day return => reward measure

    def gradient_ascent_2(self):
        n_stocks = self.get_number_of_security()
        point = self.get_equal_points(n_stocks)

        # step 1 find var for the current point
        point_var, point_string = self.get_point_var(point)

        self.logger.info(f"Starting Point: {point} \nVaR: {point_var}\n\n")
        risky_asset = []

        tolerance = 1e-6
        weight_adjustment_rate = 0.01
        stop_condition = False
        epoch = 0

        bounded_list = BoundedList(max_size=5)  # for checking last 5 changes in VaR diffs
        bounded_list.items = list(range(5))  # init random 5 different values, so code doesnt stop on first epoch
        while not stop_condition:
            # step 2 : find VaR for all possible combinations
            inter_var_diff = []
            for i in range(n_stocks):
                inter_point = point.copy()
                inter_point[i] += weight_adjustment_rate
                # Creating a new PortfolioComposition here to then calculate value at risk
                inter_point_var, inter_point_string = self.get_point_var(inter_point)

                inter_var_diff.append(inter_point_var - point_var)

            # step 3 : Find the largest absolute difference & find add locations
            largest_diff = inter_var_diff[0]
            add_loc = 0
            diff = inter_var_diff[0]
            self.logger.debug(f"{inter_var_diff} diff list")
            for i in range(len(inter_var_diff)):
                check = inter_var_diff[i]
                if abs(check) > abs(largest_diff):
                    largest_diff = abs(check)
                    diff = check
                    add_loc = i

            # step 4 : calculate new point
            if diff > 0:
                point[add_loc] += weight_adjustment_rate
            elif diff < 0:
                # Note even tho an asset = 0.01 in print, point[add_loc] - 0.01 may be less than 0 because floating point rounding error
                if point[add_loc] - weight_adjustment_rate < 0:
                    self.logger.debug("check_0")
                    self.logger.debug(point[add_loc] - weight_adjustment_rate)
                    risky_asset.append(add_loc)  # list to store order in which asset converge to 0
                else:
                    point[add_loc] -= weight_adjustment_rate

            # update value at risk for the new composition
            point_var, point_string = self.get_point_var(point)

            # add diff to bounded_list. for checking last 5 changes in VaR diff
            bounded_list.add(diff)

            if abs(diff) < tolerance:
                self.logger.debug(f"epoch: {epoch}\nChange in VaR has converged to {abs(diff)}, approximately 0. "
                                  f"The local min is \n{point}\nThe VaR is {point_var}")
                stop_condition = True
            elif epoch == 150:
                # Update epoch limit set to n, this number was set empirically
                self.logger.info(f"epoch limit was met, VaR did not converge")
                stop_condition = True
            elif bounded_list.unique_length == 1:
                # if bounded_list.unique_length returns 1, then it means last 5 epochs nothing has changed in diff
                stop_condition = True
                self.logger.debug(f"epoch: {epoch}\nVaR did not change in last 5 epochs {abs(diff)=}")

            if epoch % 5 == 0:
                # prints every 5 steps
                self.logger.debug(f"epoch : {epoch}, Change : {diff}\nVaR: {point_var}\n point : {point}\n")

            epoch += 1

        return point_string

    # This implementation of gradient descent does not return a total number of shares for the portfolio equal to the starting, shorting is not allowed.
    # This implementation identifies the riskiest company in the portfolio

    # The order that they converge to 0 shows the ranking of riskiest asset, it DOES NOT SHOW RELATIVE MAGNITUDE of risk.
    # NOTE : In this implementation only one of the assets converges to 0 than it gets stuck because the algorithm wants to take that asset to the negatives. The most risky asset of all in the port, converges to 0 first.
    # Asset that converges first is stored in list "risky_asset"

    # I THINK THE RISK ORDER CAN BE GIVEN BY THE SIZE OF least - greatest order of assets IF THEY ALL START WITH EQUAL ALLOCATIONS once the first converges to 0, the order is just size of the remaining ? #Many trials can be run.

    # The reason for risk is unknown, is it variability of price for asset, is it price of the asset, etc? This model does not reveal this info.

    # Indicate how much the return is affected by removing an asset.

    # LABEL THESE AS ORDER OF INFLUENCE IN THE PORTFOLIO !!!!! THE ONE BROUGHT TO 0 FIRST HAS THE MOST INFLUENCE IN THE VaR!!!!
