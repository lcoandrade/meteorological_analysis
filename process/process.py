from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy import fftpack
import numpy as np

class ProcessStation():
    def __init__(self, config) -> None:
        self.config = config

        self.data = pd.read_csv(filepath_or_buffer=config['file_path'], header=0, sep=';')

    def preprocess(self):
        # Calculating DATE from YEAR and DOY
        self.data["DATE"] = pd.to_datetime(
            self.data["YEAR"] * 1000 + self.data["DOY"], format="%Y%j"
        )

        self.data = self.data.set_index('DATE')

        self.data["DAY"] = self.data.index.day
        self.data["MONTH"] = self.data.index.month

        # Droping NA
        self.data = self.data.dropna()

        # Setting variable to be used
        variables = self.config["variables"]
        if len(variables) > 1:
            self.data[self.config["variable_under_analysis"]] = self.data[
                variables
            ].mean(axis=1)
        else:
            self.data[self.config["variable_under_analysis"]] = self.data[variables[0]]

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
        fourier = fftpack.fft(filtered_data.to_numpy())

        N = len(self.data)
        freqs = fftpack.fftfreq(N)
        periods = 1 / freqs

        indices = np.argsort(np.abs(fourier))[::-1]
        top_idx = indices[indices != 0][:5]

        top_periods = np.abs(periods[top_idx])

        # Imprimir os períodos encontrados
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
        plt.title("Fourier spectrum")
        plt.xlabel("Period (time unity)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(
            fname=self.config["furier_plot_path"], format=self.config["plot_format"]
        )
        plt.close()

        return tuple(ret)

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

    def multi_decompose(self, periods):
        stl_kwargs = {"seasonal_deg": 2}
        model = MSTL(
            self.data[self.variable],
            periods=self.config["periods"],
            stl_kwargs=stl_kwargs,
        )
        res = model.fit()

        seasonal = res.seasonal  # contains all seasonal components
        trend = res.trend # contains the trend
        residual = res.resid # contains the residuals

        # set the size of the plot
        plt.rcParams["figure.figsize"] = [14, 10]
        res.plot()
        plt.savefig(
            fname=self.config["multi_decomposition_plot_path"],
            format=self.config["plot_format"],
        )
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=argparse.FileType("r"), default='configurations/process.yaml')
    parsed_args = parser.parse_args()

    config = yaml.safe_load(parsed_args.config)

    processor = ProcessStation(config)
    processor.preprocess()
    processor.fix_outliers()
    processor.plot_data()
    processor.check_monthly_trends()
    processor.check_yearly_trends()
    periods = processor.get_periods()
    processor.multi_decompose(periods)
