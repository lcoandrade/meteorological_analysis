from statsmodels.tsa.seasonal import MSTL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse

class ProcessStation():
    def __init__(self, config) -> None:
        self.config = config

        self.data = pd.read_csv(filepath_or_buffer=config['file_path'], header=0, sep=';')

    def preprocess(self):
        self.data["DATE"] = pd.to_datetime(
            self.data["YEAR"] * 1000 + self.data["DOY"], format="%Y%j"
        )

        self.data = self.data.set_index('DATE')

        self.data["DAY"] = self.data.index.day
        self.data["MONTH"] = self.data.index.month

    def plot_data(self):
        plt.plot(
            self.data[self.config["variable"]],
            label=self.config["variable"],
            color="b",
            alpha=1,
            linewidth=1.0,
        )
        plt.xlabel("Year")
        plt.ylabel("Mean Air Temperature (Cº)")
        plt.grid()
        plt.legend()
        plt.savefig(fname=self.config["data_plot_path"], format=self.config['plot_format'])
        plt.close()

    def fix_outliers(self):
        indexes = self.data.index[self.data[self.config["variable"]] == 0]

        # Window size to calculate the mean
        window_size = 5

        for index in indexes:
            # Calculate the mean with the sliding window
            start_index = max(self.data.index.min(), index - pd.Timedelta(window_size, unit="days"))
            end_index = min(self.data.index.max(), index + pd.Timedelta(window_size, unit="days"))

            data_window = self.data.T2M_MAX.loc[start_index:end_index]

            smoothed_value = data_window[data_window != 0].mean()

            # Update the outlier with the mean value
            self.data.loc[index, self.config["variable"]] = smoothed_value

    def check_monthly_trends(self):
        sns_blue = sns.color_palette(as_cmap=True)[0]

        # Defina uma figura e um conjunto de subplots com 4 linhas e 3 colunas
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))

        # Agrupe os dados por mês e dia, calcule a média para cada grupo e reorganize os dados
        monthly_mean = (
            self.data.groupby(["MONTH", "DAY"])[self.config["variable"]]
            .mean()
            .unstack()
        )

        # Crie um loop para iterar pelos meses (de 01 a 12)
        for i, month in enumerate(range(1, 13)):
            # Calcule as coordenadas do subplot atual
            row = i // 3
            col = i % 3

            # Filtrar os dados para o mês atual
            data_for_month = self.data[self.data.MONTH == month]

            # Plotar os dados no subplot correspondente
            axs[row, col].plot(
                data_for_month.DAY,
                data_for_month[self.config["variable"]],
                label=self.config["variable"],
                color=sns_blue,
                alpha=0.1,
            )
            axs[row, col].set_title(f"Month {str(month)}")
            axs[row, col].legend()

            # Selecione a média mensal para o mês atual
            monthly_data = monthly_mean.loc[month]

            # Plotar a média mensal no subplot correspondente
            axs[row, col].plot(
                monthly_data,
                label=f"Mean T2M_MAX",
                color="blue",
                alpha=1,
            )
            axs[row, col].legend()

        # Ajustar o layout e rótulos
        plt.tight_layout()
        plt.savefig(fname=self.config["monthly_trends_plot_path"], format=self.config['plot_format'])
        plt.close()

    def check_yearly_trends(self):
        sns_blue = sns.color_palette(as_cmap=True)[0]

        # Obtenha a lista de anos únicos no DataFrame
        unique_years = self.data["YEAR"].unique()

        # Defina o número de subplots com base na quantidade de anos únicos
        num_subplots = len(unique_years)

        # Calcule o número de linhas e colunas com base no número de subplots desejados
        num_cols = 3
        num_rows = num_subplots // num_cols

        # Crie a figura e os subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 30))

        # Flatten a matriz de subplots para simplificar o loop
        axs = axs.flatten()

        # Agrupe os dados por ano e mês, calcule a média para cada grupo e reorganize os dados
        yearly_mean = (
            self.data.groupby(["YEAR", "MONTH"])[self.config["variable"]]
            .mean()
            .unstack()
        )

        # Crie um loop para iterar pelos anos
        for i, year in enumerate(unique_years[: num_cols * num_rows]):
            # Filtrar os dados para o mês atual
            data_for_year = self.data[self.data.YEAR == year]

            # Plotar os dados no subplot correspondente
            axs[i].plot(
                data_for_year.MONTH,
                data_for_year[self.config["variable"]],
                label=self.config["variable"],
                color=sns_blue,
                alpha=0.1,
            )
            axs[i].set_title(f"Year {str(year)}")
            axs[i].legend()

            # Selecione a média mensal para o mês atual
            yearly_data = yearly_mean.loc[year]

            # Plotar a média mensal no subplot correspondente
            axs[i].plot(
                yearly_data,
                label=f"Mean T2M_MAX",
                color="blue",
                alpha=1,
            )
            axs[i].legend()

        # Ajustar o layout e rótulos
        plt.tight_layout()
        plt.savefig(fname=self.config["yearly_trends_plot_path"], format=self.config['plot_format'])
        plt.close()

    def series_decomposition(self):
        stl_kwargs = {"seasonal_deg": 0}
        model = MSTL(
            self.data[self.config["variable"]],
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
        plt.savefig(fname=self.config["decomposition_plot_path"], format=self.config['plot_format'])

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
    processor.series_decomposition()
