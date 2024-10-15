from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy import fft
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import nolds
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress

# Multi run using hydra
@hydra.main(
    version_base=None, config_path="../configurations", config_name="default_station"
)
def multi_run(cfg: DictConfig):
    """
    Executes multiple yaml files using Hydra
    """
    print(OmegaConf.to_yaml(cfg))
    processor = ProcessStation(config=cfg, multi=True)
    processor.main()


# Single run using yaml configuration
def single_run():
    """
    Performs a single yaml execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        default="configurations/stations/el_carmen.yaml",
    )
    parsed_args = parser.parse_args()

    config = yaml.safe_load(parsed_args.config)

    processor = ProcessStation(config, multi=False)
    processor.main()


class ProcessStation:
    def __init__(self, config, multi) -> None:
        if multi:
            self.config = config["stations"]
        else:
            self.config = config

        self.data = pd.read_csv(
            filepath_or_buffer=self.config["file_path"],
            header=0,
            sep=self.config["separator"],
            decimal=self.config["decimal"],
            skiprows=15,
        )

    def preprocess(self):
        """
        Preprocess data according to its yaml file
        """
        if self.config["date_format"] == "%Y%j":
            # Calculating DATE from YEAR and DOY
            self.data["DATE"] = pd.to_datetime(
                self.data["YEAR"] * 1000 + self.data["DOY"],
                format=self.config["date_format"],
            )
            # Setting date as index and calculating day and month for group by purposes
            self.data = self.data.set_index("DATE")
            self.data["DAY"] = self.data.index.day
            self.data["MONTH"] = self.data.index.month
        else:
            self.data["DATE"] = pd.to_datetime(
                self.data["DATE"], format=self.config["date_format"]
            )
            # Setting date as index and calculating day and month for group by purposes
            self.data = self.data.set_index("DATE")
            self.data["DAY"] = self.data.index.day
            self.data["MONTH"] = self.data.index.month
            self.data["YEAR"] = self.data.index.year

        # Droping NA
        self.data = self.data.dropna()

        # Setting variable to be used
        variables = self.config["variables"]
        if len(variables) > 1:
            self.data[self.config["variable_under_analysis"]] = self.data[
                variables
            ].mean(axis=1)

        self.variable = self.config["variable_under_analysis"]

    def plot_data(self):
        """
        Plots the data and save a PDF file in the plots folder
        """
        plt.plot(
            self.data[self.variable],
            label=self.variable,
            color="b",
            alpha=1,
            linewidth=1.0,
        )
        plt.xlabel("Year")
        plt.ylabel(self.variable)
        plt.grid()
        plt.title(f"Station: {self.config['station']}")
        plt.legend()
        plt.savefig(fname=self.config["data_plot_path"], format=self.config['plot_format'])
        plt.close()

    def fix_outliers(self):
        """
        Fix outliers with a specific value
        """
        indexes = self.data.index[self.data[self.variable] == -999.0]

        # Window size to calculate the mean
        window_size = 5

        for index in indexes:
            # Calculate the mean with the sliding window
            start_index = max(self.data.index.min(), index - pd.Timedelta(window_size, unit="days"))
            end_index = min(self.data.index.max(), index + pd.Timedelta(window_size, unit="days"))

            data_window = self.data[self.variable].loc[start_index:end_index]

            smoothed_value = data_window[data_window != 0].mean()

            # Update the outlier with the mean value
            self.data.loc[index, self.variable] = smoothed_value

    def check_monthly_trends(self):
        """
        Check if the data presents monthly trends.
        The data across all years are group by month and day.
        Then, for each month of the year, the dayly mean in plotted over the day data.
        """
        sns_blue = sns.color_palette(as_cmap=True)[0]

        # Define a figure and a set of subplots with 4 lines and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))

        # Group the data by month and day, calculate the mean for each group and reorganize de data
        monthly_mean = (
            self.data.groupby(["MONTH", "DAY"])[self.variable].mean().unstack()
        )

        # Loop over the months (1 to 12)
        for i, month in enumerate(range(1, 13)):
            # Calculate the coordinates of the current subplot
            row = i // 3
            col = i % 3

            # Filtering the data for the current month
            data_for_month = self.data[self.data.MONTH == month]

            # Plot the data in the correspondent subplot
            axs[row, col].plot(
                data_for_month.DAY,
                data_for_month[self.variable],
                label=self.variable,
                color=sns_blue,
                alpha=0.1,
            )
            axs[row, col].set_title(f"Month {str(month)}")
            axs[row, col].legend()

            # Select the monthly mean for the current month
            monthly_data = monthly_mean.loc[month]

            # Plot the monthly mean in the correspondent subplot
            axs[row, col].plot(
                monthly_data,
                label=f"Mean {self.variable}",
                color="blue",
                alpha=1,
            )
            axs[row, col].legend()

        # Set layout and labels
        plt.tight_layout()
        plt.savefig(fname=self.config["monthly_trends_plot_path"], format=self.config['plot_format'])
        plt.close()

    def check_yearly_trends(self):
        """
        Check if the data presents yearly trends.
        The data across all years are group by year and month.
        Then, for each year, the monthly mean in plotted over the month data.
        """
        sns_blue = sns.color_palette(as_cmap=True)[0]

        # Get the unique years from the data frame
        unique_years = self.data["YEAR"].unique()

        # Define the subplots number
        num_subplots = len(unique_years)

        # Calculate rows and cols to plot
        num_cols = 3
        num_rows = num_subplots // num_cols

        # Create figure and subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 30))

        # Flatten the subplots matrix to simplify the loop
        axs = axs.flatten()

        # Group the data by month and day, calculate the mean for each group and reorganize de data
        yearly_mean = (
            self.data.groupby(["YEAR", "MONTH"])[self.variable].mean().unstack()
        )

        # Loop over the year
        for i, year in enumerate(unique_years[: num_cols * num_rows]):
            # Filtering the data for the current month
            data_for_year = self.data[self.data.YEAR == year]

            # Plot the data in the correspondent subplot
            axs[i].plot(
                data_for_year.MONTH,
                data_for_year[self.variable],
                label=self.variable,
                color=sns_blue,
                alpha=0.1,
            )
            axs[i].set_title(f"Year {str(year)}")
            axs[i].legend()

            # Select the yearly mean for the current month
            yearly_data = yearly_mean.loc[year]

            # Plot the yearly mean in the correspondent subplot
            axs[i].plot(
                yearly_data,
                label=f"Mean {self.variable}",
                color="blue",
                alpha=1,
            )
            axs[i].legend()

        # Set layout and labels
        plt.tight_layout()
        plt.savefig(fname=self.config["yearly_trends_plot_path"], format=self.config['plot_format'])
        plt.close()

    def get_periods(self):
        """
        Perform a FFT on the series to determine the top 5 meaningful periods

        returns
            ret: list[int]
                List of top 5 periods in unities of time
        """
        filtered_data = self.data[self.variable]

        # Calculate the fft of the data
        # Return complex values
        fourier = fft.rfft(filtered_data.to_numpy())

        # Calculating the frequencies
        N = len(self.data)
        freqs = fft.rfftfreq(N)
        periods = 1 / freqs

        # Getting the frequencies with higher magnitudes
        indices = np.argsort(np.abs(fourier))[::-1]
        # Getting the top five frequency indexes
        top_idx = indices[indices != 0][:5]

        # Calculating the top 5 periods
        top_periods = np.abs(periods[top_idx])

        # Printing the periods
        print("Found periods:")
        ret = []
        for i, period in enumerate(top_periods):
            value = int(period)
            # Not keeping the Nyquist frequency
            if value not in ret and value != N:
                ret.append(value)
                print(f"Peak {i+1}: {value} unities of time")

        # Plot the Furier spectrum for visualizations purposes
        plt.figure(figsize=(10, 6))
        plt.plot(periods, np.abs(fourier))
        plt.title(f"Fourier spectrum (Station: {self.config['station']})")
        plt.xlabel("Period (time unity)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(
            fname=self.config["furier_plot_path"], format=self.config["plot_format"]
        )
        plt.close()

        return ret

    def multi_decompose(self, periods):
        """
        Performs MSTL decomposition on the time series
        Saves a PDF graphic with trend, seasons and noise
        """
        stl_kwargs = {"seasonal_deg": 2}
        model = MSTL(
            self.data[self.variable],
            periods=periods,
            stl_kwargs=stl_kwargs,
        )
        res = model.fit()

        # seasonal = res.seasonal  # contains all seasonal components
        trend = res.trend  # contains the trend
        # residual = res.resid # contains the residuals

        # set the size of the plot
        plt.rcParams["figure.figsize"] = [14, 10]
        res.plot()
        plt.savefig(
            fname=self.config["multi_decomposition_plot_path"],
            format=self.config["plot_format"],
        )
        plt.close()

        # Linear regression on the trend to check warmth or cooling
        y = trend.values
        x = np.arange(len(y))
        result = linregress(x=x, y=y)

        # Plotting the linear regression
        self.plot_trend_regression(x, y, result)

        return "warmth" if result.slope > 0 else "cooling"

    def plot_trend_regression(self, x, y, result):
        """
        Plots the trend regression to analyze wether there is a warmth trend

        params:
            x: trend x-values as numpy array
            y: trend y-values as numpy array
            result: LinregressResult instance
        """

        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            x,
            x * result.slope + result.intercept,
            label=f"linear regression (Slope:{result.slope})",
            color="r",
        )
        plt.plot(
            x,
            y,
            label="Trend",
            color="b",
        )
        plt.title(f"Trend's linear regression (Station: {self.config['station']})")
        plt.xlabel("Days")
        plt.ylabel(self.variable)
        plt.legend()
        plt.savefig(
            fname=self.config["trend_regression_ploth_path"],
            format=self.config["plot_format"],
        )
        plt.close()

    def plot_exponents(self, x, y, method, xlabel, path):
        """
        Plots the multiple values of the computed exponents to check the series exponent behavior
        """
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            x,
            y,
            label=f"{method}",
        )
        plt.title(f"{method} (Station: {self.config['station']})")
        plt.xlabel(xlabel)
        plt.ylabel("Exponent")
        plt.legend()
        plt.savefig(fname=path, format=self.config["plot_format"])
        plt.close()

    def check_stationarity(self, confidence_interval=0.05):
        """
        Executes de Augmented Dickey-Fuller test to check for stationarity in time series
            Null Hypothesis (H0):
                If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary.
                It has some time dependent structure.
            Alternate Hypothesis (H1):
                The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary.
                It does not have time-dependent structure.
        parameters
            confidence_interval: float
                Confidence Interval for the test
        returns
            boolean:
                True if the series is stationary
                False if the series is non-stationary
        """
        data = self.data[self.variable]
        result = adfuller(data)
        p_value = result[1]
        if p_value > confidence_interval:
            # Fail to reject H0. The data has a unit root and is non-stationary.
            return False
        else:
            # Reject H0 (accept H1). The data does not have a unit root and is stationary.
            return True

    def compute_ltm(self):
        """
        Computes Long-Term Memory (LTM) exponents using different methods:
            Hurst Rescaled Range (R/S)
            Detrended Fluctuation Analysis (DFA)

        returns
            ltm_dict: dictionary
                Dictionary with max and min hurst values for each method
        """
        data = self.data[self.variable]

        ltm_dict = {}

        # Total data size
        total = len(data)

        # Calculating LTM using Hurst exponent (hurst_rs)
        # Calculating a series of nvals of powers of 2
        # max_power is the maximum k such that 2ˆk <= total/2 (i.e. 2ˆk is the maximum subseries size)
        max_power = int(np.log2(total / 2))
        # levels of subdivision (i.e. level 4 means 4 subdivisions in powers of 2)
        levels = np.arange(3, max_power)
        # Subseries sizes (i.e. list of 2ˆk)
        subserie_sizess = [
            2 ** np.arange(max_power - level, max_power + 1) for level in levels
        ]

        hurst_rs = [
            nolds.hurst_rs(data, nvals=subserie_sizes, fit="poly")
            for subserie_sizes in subserie_sizess
        ]
        ltm_dict["rs_max"] = max(hurst_rs)
        ltm_dict["rs_min"] = min(hurst_rs)
        # Plotting the data
        self.plot_exponents(
            levels,
            hurst_rs,
            method="Hurst R/S",
            xlabel="Divisor factor k (i.e. 2ˆk)",
            path=self.config["hurst_rs_plot_path"],
        )

        # Calculating LTM using detrended fluctuation analysis (DFA)
        alpha_dfa = [
            nolds.dfa(data, nvals=subserie_sizes, fit_trend="poly", fit_exp="RANSAC")
            for subserie_sizes in subserie_sizess
        ]
        ltm_dict["dfa_max"] = max(alpha_dfa)
        ltm_dict["dfa_min"] = min(alpha_dfa)
        # Plotting the data
        self.plot_exponents(
            levels,
            alpha_dfa,
            method="Alpha DFA",
            xlabel="Divisor factor k (i.e. 2ˆk)",
            path=self.config["hurst_dfa_plot_path"],
        )

        return ltm_dict

    def get_mean_period(self):
        """
        Calculates de mean period of a time series to use in the Lyapunov exponent calculation using FFT
        Based on nolds: https://github.com/CSchoel/nolds/blob/main/nolds/measures.py#L247

        params
            max_tsep_factor: float
                Factor to used in the min_tsep calculation
        returns
            min_tsep: int
                Mean time series period
        """
        data = self.data[self.variable]
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        # max_tsep_factor = 0.25
        f = np.fft.rfft(data, n * 2 - 1)
        mf = np.fft.rfftfreq(n * 2 - 1) * f**2
        mf = np.sum(mf[1:]) / np.sum(f[1:] ** 2)

        min_tsep = np.ceil(1 / mf.real)
        # min_tsep = min(min_tsep, int(max_tsep_factor * n))
        # return int(min_tsep)
        return min_tsep

    def compute_lyapunov(self, periods):
        """
        Estimates the largest Lyapunov exponent using the algorithm of Rosenstein

        params
            periods: list of ints
                Top 5 FFT periods present in  the series

        returns
            lyap_dict: dictionary
                Dictionary with max and min lyap_r values
        """
        lyap_dict = {}

        # Also testing the mean period as stated by
        # Rosenstein, M. T., Collins, J. J., & de Luca, C. J. (1993).
        # A practical method for calculating largest Lyapunov exponents from small data sets.
        # Physica D: Nonlinear Phenomena, 65(1–2), 117–134. https://doi.org/10.1016/0167-2789(93)90009-P
        periods.append(int(np.mean(periods)))
        periods.sort()

        lyap_r = [
            nolds.lyap_r(list(self.data[self.variable]), min_tsep=period, fit="poly")
            for period in periods
        ]

        # Plotting the data
        self.plot_exponents(
            periods,
            lyap_r,
            method="Lyapunov (Rosenstein's)",
            xlabel="Period",
            path=self.config["lyapunov_plot_path"],
        )
        lyap_dict["lyap_max"] = max(lyap_r)
        lyap_dict["lyap_min"] = min(lyap_r)

        return lyap_dict

    def save_report(self, periods, ltm_dict, lyap_dict, change):
        """
        Saves a CSV report with all values calculated

        params
            periods: list
            ltm_dict: dictionary
            adfuller: boolean
            lyap_r: float
        """
        report = {
            "Station": self.config["station"],
            "Periods": [periods],
            "Hurst (R/S) max": ltm_dict["rs_max"],
            "Hurst (R/S) min": ltm_dict["rs_min"],
            "Alpha (DFA) max": ltm_dict["dfa_max"],
            "Alpha (DFA) min": ltm_dict["dfa_min"],
            "Lyapunov (Rosenstein's) max": lyap_dict["lyap_max"],
            "Lyapunov (Rosenstein's) min": lyap_dict["lyap_min"],
            "Trend": change,
        }

        # Saving the global report file
        report_file = "process/report.csv"
        new_df = pd.DataFrame.from_dict(
            report,
        )
        if Path(report_file).is_file():
            df = pd.read_csv(report_file)
            df = df._append(new_df)
        else:
            df = new_df
        df.to_csv(report_file, index=False)

    def main(self):
        """
        Executes the full process
        """
        self.preprocess()
        self.fix_outliers()
        self.plot_data()
        self.check_monthly_trends()
        self.check_yearly_trends()
        periods = self.get_periods()
        change = self.multi_decompose(periods[:3])
        ltm_dict = self.compute_ltm()
        lyap_dict = self.compute_lyapunov(periods)
        self.save_report(periods, ltm_dict, lyap_dict, change)


if __name__ == "__main__":
    multi_run()
    # single_run()
