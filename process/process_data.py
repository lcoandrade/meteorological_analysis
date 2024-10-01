from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from scipy import fftpack
import numpy as np
import nolds
from pathlib import Path


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
        plt.savefig(
            fname=self.config["data_plot_path"], format=self.config["plot_format"]
        )
        plt.close()

    def fix_outliers(self):
        indexes = self.data.index[self.data[self.variable] == 0]

        # Window size to calculate the mean
        window_size = 5

        for index in indexes:
            # Calculate the mean with the sliding window
            start_index = max(
                self.data.index.min(), index - pd.Timedelta(window_size, unit="days")
            )
            end_index = min(
                self.data.index.max(), index + pd.Timedelta(window_size, unit="days")
            )

            data_window = self.data[self.variable].loc[start_index:end_index]

            smoothed_value = data_window[data_window != 0].mean()

            # Update the outlier with the mean value
            self.data.loc[index, self.variable] = smoothed_value

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
            nolds.hurst_rs(
                self.data[self.variable],
                nvals=nvals,
                fit="poly",
                debug_plot=True,
                plot_file=f"plots/{self.config['station']}_hurst_rs_{i}.pdf",
            )
            for i, nvals in enumerate(nvalss)
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
            nolds.dfa(
                self.data[self.variable],
                nvals=nvals,
                fit_exp="poly",
                debug_plot=True,
                plot_file=f"plots/{self.config['station']}_hurst_dfa_{i}.pdf",
            )
            for i, nvals in enumerate(nvalss)
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
            nolds.mfhurst_b(
                self.data[self.variable],
                dists=dists,
                debug_plot=True,
                plot_file=f"plots/{self.config['station']}_hurst_ghe_{i}.pdf",
            )
            for i, dists in enumerate(distss)
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

    def run_mf(self):
        data = self.data[self.variable]
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        max_tsep_factor = 0.25
        f = np.fft.rfft(data, n * 2 - 1)
        mf = np.fft.rfftfreq(n * 2 - 1) * f**2
        mf = np.sum(mf[1:]) / np.sum(f[1:] ** 2)

        print("MF: ", mf)
        print("MF Type:", type(mf))

    def main(self):
        self.preprocess()
        self.fix_outliers()
        # periods = self.get_periods()
        # hurst_dict = self.compute_Hurst()
        # lyap_r = self.compute_Lyapunov()
        # lyap_r = 0
        # self.save_report(periods, hurst_dict, lyap_r)
        self.run_mf()


if __name__ == "__main__":
    # multi_run()
    single_run()
