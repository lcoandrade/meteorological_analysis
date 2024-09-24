## meteorological_analysis
meteorological_analysis provides a Python code that allows:
1. Process meteorological data is CSV format in order to determine periods and trends
2. It uses:
   - FFT to determine periods
   - Hurst's Coefficient to quantify long-term memory of a time series to determine persistent/anti-persistent/random-walk behaviors
   - Lyapunov's exponent to measure the sensitivity of a time series to its initial conditions to determine chaotic/stable behaviors
   - MSTL to decompose the time series in trend, seasonality and residual, considering a persistence and stable behavior 

## PIP Requirements
1. hydra: to run multiple configuration files
2. nolds: to compute Lyapunov's exponent
3. hurst: to compute Hurst's coefficient
4. statsmodels: to decompose series with MSTL
5. Basic dependencies: Matplotlib, Scikit-learn, Numpy, Pandas 

## Processing all configuration files with multirun (run in the base folder)
python process/process.py -m stations=el_carmen,inaquito,peribuela,pomasqui,puerto_ila,tabacundo,tola,tumbaco