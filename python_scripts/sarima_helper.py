from statsmodels.tsa.stattools import adfuller
import numpy as np

def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']
    if (pvalue < 0.05) and (adf < critical_value):
        return True
    else:
        return False

def differentiate_untill_stationary(ts):
    d = 0
    ts_diff = ts
    while not check_stationarity(ts_diff):
        ts_diff = ts.diff()
        ts_diff.dropna(inplace=True)

    return d

