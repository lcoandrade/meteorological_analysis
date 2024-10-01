from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy import fftpack
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hurst import compute_Hc
import nolds
from pathlib import Path


# Multi run using hydra
@hydra.main(
    version_base=None, config_path="../configurations", config_name="default_station"
)
def multi_run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    processor = ProcessStation(config=cfg, multi=True)
    processor.main()


# Single run using yaml configuration
def single_run():
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


class ProcessStation():

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
        )

    def preprocess(self):
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
        indexes = self.data.index[self.data[self.variable] == 0]

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
        filtered_data = self.data[self.variable]

        # Calculate the fft of the data
        # Return complex values
        fourier = fftpack.fft(filtered_data.to_numpy())

        # Calculating the frequencies
        N = len(self.data)
        freqs = fftpack.fftfreq(N)

        # Nyquist frequency index
        nyquist_idx = N // 2

        # Focusing on the positive part of the spectrum
        freqs = freqs[: nyquist_idx + 1]
        fourier = fourier[: nyquist_idx + 1]

        # Getting the frequencies with higher magnitudes
        indices = np.argsort(np.abs(fourier))[::-1]
        # Getting the top five frequency indexes
        top_idx = indices[indices != 0][:5]

        # Calculating the top 5 periods
        periods = 1 / freqs
        top_periods = np.abs(periods[top_idx])

        # Printing the periods
        print("Found periods:")
        ret = []
        for i, period in enumerate(top_periods):
            value = int(period)
            if value not in ret and value < N // 2:
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

    def single_decompose(self, period):
        results = seasonal_decompose(
            x=self.data[self.variable],
            model="additive",
            period=period,
        )

        plt.figure(figsize=(12, 10))
        plt.subplot(411)
        plt.plot(self.data[self.variable], label="Series")
        plt.legend(loc="best")
        plt.subplot(412)
        plt.plot(results.trend, label="Trend")
        plt.legend(loc="best")
        plt.subplot(413)
        plt.plot(results.seasonal, label="Seasonal")
        plt.legend(loc="best")
        plt.subplot(414)
        plt.plot(results.resid, label="Residual")
        plt.legend(loc="best")
        plt.savefig(
            fname=self.config["single_decomposition_plot_path"],
            format=self.config["plot_format"],
        )
        plt.close()

    def multi_decompose(self):
        stl_kwargs = {"seasonal_deg": 2}
        model = MSTL(
            self.data[self.variable],
            periods=self.config["periods"],
            stl_kwargs=stl_kwargs,
        )
        res = model.fit()

        # seasonal = res.seasonal  # contains all seasonal components
        # trend = res.trend # contains the trend
        # residual = res.resid # contains the residuals

        # set the size of the plot
        plt.rcParams["figure.figsize"] = [14, 10]
        res.plot()
        plt.savefig(
            fname=self.config["multi_decomposition_plot_path"],
            format=self.config["plot_format"],
        )
        plt.close()

    def plot_hurst(self, x, y, method, xlabel, path):
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            x,
            y,
            label=f"Hurst {method}",
        )
        plt.title(f"Hurst {method} (Station: {self.config['station']})")
        plt.xlabel(xlabel)
        plt.ylabel("Hurst exponent")
        plt.legend()
        plt.savefig(fname=path, format=self.config["plot_format"])
        plt.close()

    def compute_Hurst(self):
        hurst_dict = {}

        # Computing Hurst using nolds package
        total = len(self.data[self.variable])

        # Calculating Hurst using Hurst exponent (hurst_rs)
        nstepss = np.arange(15, 31)
        nvalss = [
            nolds.logmid_n(total, ratio=1 / 4.0, nsteps=nsteps) for nsteps in nstepss
        ]
        hurst_rs = [
            nolds.hurst_rs(self.data[self.variable], nvals=nvals, fit="poly")
            for nvals in nvalss
        ]
        hurst_dict["rs_max"] = max(hurst_rs)
        hurst_dict["rs_min"] = min(hurst_rs)
        # Plotting the data
        self.plot_hurst(
            nstepss,
            hurst_rs,
            method="R/S",
            xlabel="Subserie size",
            path=self.config["hurst_rs_plot_path"],
        )

        # Calculating Hurst using detrended fluctuation analysis (DFA) (dfa)
        factors = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        nvalss = [nolds.logarithmic_n(4, 0.1 * total, factor) for factor in factors]
        hurst_dfa = [
            nolds.dfa(self.data[self.variable], nvals=nvals, fit_exp="poly")
            for nvals in nvalss
        ]
        hurst_dict["dfa_max"] = max(hurst_dfa)
        hurst_dict["dfa_min"] = min(hurst_dfa)
        # Plotting the data
        self.plot_hurst(
            factors,
            hurst_dfa,
            method="DFA",
            xlabel="Factor",
            path=self.config["hurst_dfa_plot_path"],
        )

        # Calculating Hurst using Generalized Hurst Exponent (mfhurst_b)
        factors = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        distss = [
            nolds.logarithmic_n(1, max(20, 0.02 * total), factor) for factor in factors
        ]
        hurst_ghe = [
            nolds.mfhurst_b(self.data[self.variable], dists=dists) for dists in distss
        ]
        hurst_dict["ghe_max"] = max(hurst_ghe)
        hurst_dict["ghe_min"] = min(hurst_ghe)
        # Plotting the data
        self.plot_hurst(
            factors,
            hurst_ghe,
            method="GHE",
            xlabel="Factor",
            path=self.config["hurst_ghe_plot_path"],
        )

        # Calculating Hurst using Ernest Chan's principles
        n_lagss = np.arange(30, 1460, 30)
        hurst_chan = [self.hurst_ernest_chan(n_lags) for n_lags in n_lagss]
        hurst_dict["chan_max"] = max(hurst_chan)
        hurst_dict["chan_min"] = min(hurst_chan)
        # Plotting the data
        self.plot_hurst(
            n_lagss,
            hurst_chan,
            method="CHAN",
            xlabel="N_lags",
            path=self.config["hurst_chan_plot_path"],
        )

        return hurst_dict

    def hurst_ernest_chan(self, n_lags):
        """Returns the Hurst Exponent of the time series vector ts"""
        ts = list(self.data[self.variable])

        # Create the range of lag values
        lags = range(2, n_lags)

        # Calculate the array of the variances of the lagged differences
        # This means log(std) = 0.5*log(var)
        # Therefore, the exponent of polyfit needs to be multiplied by 2 to give H
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    def compute_Lyapunov(self):
        lyap_r = nolds.lyap_r(list(self.data[self.variable]), min_tsep=1)
        # lyap_e = nolds.lyap_e(self.data[self.variable])

        return lyap_r

    def save_report(self, periods, hurst_dict, lyap_r):
        report = {
            "Station": self.config["station"],
            "Periods": [periods],
            "Hurst (R/S) max": hurst_dict["rs_max"],
            "Hurst (R/S) min": hurst_dict["rs_min"],
            "Hurst (DFA) max": hurst_dict["dfa_max"],
            "Hurst (DFA) min": hurst_dict["dfa_min"],
            "Hurst (GHE) max": hurst_dict["ghe_max"],
            "Hurst (GHE) min": hurst_dict["ghe_min"],
            "Hurst (CHAN) max": hurst_dict["chan_max"],
            "Hurst (CHAN) min": hurst_dict["chan_min"],
            "Lyapunov (Rosenstein's)": lyap_r,
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
        self.preprocess()
        self.fix_outliers()
        self.plot_data()
        self.check_monthly_trends()
        self.check_yearly_trends()
        periods = self.get_periods()
        self.multi_decompose()
        hurst_dict = self.compute_Hurst()
        lyap_r = self.compute_Lyapunov()
        # lyap_r = 0
        self.save_report(periods, hurst_dict, lyap_r)


if __name__ == "__main__":
    multi_run()
    # single_run()
