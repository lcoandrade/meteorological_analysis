## meteorological_analysis
meteorological_analysis provides a Python code that allows:
1. Process meteorological data is CSV format in order to determine periods and trends
2. It uses:
   - FFT to determine periods
   - Hurst's Coefficient to quantify long-term memory of a time series to determine persistent/anti-persistent/random-walk behaviors
   - Lyapunov's exponent to measure the sensitivity of a time series to its initial conditions to determine chaotic/stable behaviors
   - MSTL to decompose the time series in trend, seasonality and residual, considering a persistence and stable behavior 

## Fourier Transforms
[Fourier Transforms]{https://docs.scipy.org/doc/scipy/tutorial/fft.html#d-discrete-fourier-transforms}

## Hurst coefficient
[Hurst Exponent]{https://en.wikipedia.org/wiki/Hurst_exponent}
References from NOLDS:
    H. E. Hurst, “The problem of long-term storage in reservoirs,”
       International Association of Scientific Hydrology. Bulletin, vol. 1,
       no. 3, pp. 13–27, 1956.
    H. E. Hurst, “A suggested statistical model of some time series
       which occur in nature,” Nature, vol. 180, p. 494, 1957.
    R. Weron, “Estimating long-range dependence: finite sample
       properties and confidence intervals,” Physica A: Statistical Mechanics
       and its Applications, vol. 312, no. 1, pp. 285–299, 2002.

## Lyapunov coefficient
[Lyapunov coefficient]{https://en.wikipedia.org/wiki/Lyapunov_exponent}
References from NOLDS:
    M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
       “A practical method for calculating largest Lyapunov exponents from
       small data sets,” Physica D: Nonlinear Phenomena, vol. 65, no. 1,
       pp. 117–134, 1993.

## PIP Requirements
1. hydra: to run multiple configuration files
2. nolds: to compute Lyapunov's exponent
3. statsmodels: to decompose series with MSTL
4. Basic dependencies: Matplotlib, Scikit-learn, Numpy, Pandas 

## Processing all configuration files with multirun (run in the base folder)
python process/process.py -m stations=el_carmen,inaquito,peribuela,pomasqui,puerto_ila,tabacundo,tola,tumbaco