import itertools
import multiprocessing
import warnings
from math import sqrt

import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
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




def mape(actual,pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def create_lagged_value_columns(lag_features, windows, df):
    df.reset_index(drop=True, inplace=True)
    for window in windows:
        for feature in lag_features:
            try:
                df[f"{feature}_mean_lag{window}"] = df[feature].shift(window)
            except:
                pass

    df.fillna(df.mean(), inplace=True)
    df.set_index(["Date"], drop=False, inplace=True)
    return df

def predict_with_windows(windows, train, test, lag_features, is_advanced=False):
    for window in windows:
        exogs= []
        for feature in lag_features:
            try:
                if f"{feature}_mean_lag{window}" in train.columns:
                    exogs.append(f"{feature}_mean_lag{window}")
                if is_advanced and f"{feature}_std_lag{window}" in train.columns:
                    exogs.append(f"{feature}_std_lag{window}")
            except:
                print(f'There is no {feature}_mean_lag{window} in data frame')

        model = auto_arima(train.Close, exogenous=train[exogs], trace=True, error_action="ignore",
                           suppress_warnings=True)
        model.fit(train.Close, exogenous=train[exogs])

        forecast = model.predict(n_periods=len(test), exogenous=test[exogs])
        test[f"Forecast_ARIMAX_{window}d_lag"] = forecast
        test[["Close", f"Forecast_ARIMAX_{window}d_lag"]].plot(figsize=(17, 9))
        print(f"MAPE for {window} day lag prediction is {mape(test['Close'], forecast)}")


def remove_lagged_features_with_window_time(window, features_to_remove, df, lag_features):
    for feature in features_to_remove:
        try:
            del df[f"{feature}_mean_lag{window}"]
            lag_features.remove(feature)
        except:
            pass


def train_test_split_continual(df, test_percentage):
    testNum = round(df.shape[0] * test_percentage)
    train = df.iloc[:-testNum]
    test = df.iloc[-testNum:]

    return train, test




