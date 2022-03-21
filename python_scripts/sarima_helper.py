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

class ArimaModelStats:
    def __init__(self, p, d, q, aic, bic, rmse, predicted, actual, mape=0):
        self.p = p
        self.q = q
        self.d = d
        self.rmse = rmse
        self.aic = aic
        self.bic = bic
        self.predicted = predicted
        self.actual = actual
        self.mape = mape

    def draw_graph(self):
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(self.actual, 'blue', label='Test data')
        plt.plot(self.predicted, 'green', label='Predicted')
        plt.legend()
        plt.show()

    def explain_model(self):
        print(f"ARIMA model ({self.p} {self.d} {self.q}) "
              f"AIC={self.aic} BIC={self.bic} RMSE={self.rmse} MAPE={self.mape}\n")

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


def try_different_model_combinations(p_range, q_range, d_range, df_train, df_test, prediction_variable,
                                     lag_features, windows):
    p_s = p_range
    q_s = q_range
    d_s = d_range

    processor_count = 6

    all_model_combinations = itertools.product(p_s, d_s, q_s)
    all_model_combinations = np.array(list(all_model_combinations))

    process_pool = multiprocessing.Pool(processor_count)
    array = np.array_split(all_model_combinations, processor_count)

    for window in windows:
        exogs= []
        best_models = []
        process_results = []
        for feature in lag_features:
            exogs.append(f"{feature}_mean_lag{window}")
        for i in range(0, processor_count):
            process_results.append(process_pool.apply_async(test_ARIMA_model_combinations,
                                                            args=(array[i], df_train, prediction_variable,
                                                                  df_test, i, exogs)))
            for result in process_results:
                result_proc = result.get()
                print(result_proc)
                best_models.extend(result_proc)
                print('Process finished!')

            best_models.sort(key=lambda x: (x.aic, x.rmse, x.bic, x.mape))
            best_model = best_models[0]
            print(f"BEST MODEL for window {window}")
            best_model.draw_graph()
            best_model.explain_model()


def test_ARIMA_model_combinations(combinations, df_train, prediction_variable, df_test, proc_id, exog_vars):
    models = []
    for combination in combinations:
        print(f"{proc_id}----testing model {combination[0]}, {combination[1]}, {combination[2]} ")

        p = combination[0]
        d = combination[1]
        q = combination[2]

        try:
            model = SARIMAX(df_train[prediction_variable], order=(p, d, q), exog=df_train[exog_vars] )
            model = model.fit()
            start = len(df_train)
            end = len(df_train) + len(df_test) - 1
            pred = model.predict.predict(start=start, end=end, exog=df_test[exog_vars],  typ='levels').rename('ARIMA predictions')
            rmse = sqrt(mean_squared_error(df_test.Close, pred))
            mape_c = mape(df_test.Close,pred)
            models.append(ArimaModelStats(p, d, q, abs(model.aic),
                                               abs(model.bic), rmse, pred, df_test.Close, mape=mape_c))

        except:
            print('Model failed with error, resuming..')
    print(f'---process {proc_id} has finished---')
    return models

def mape(actual,pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def create_lagged_value_columns(lag_features, windows, df):
    df.reset_index(drop=True, inplace=True)
    for window in windows:
        for feature in lag_features:
            df[f"{feature}_mean_lag{window}"] = df[feature].shift(window)

    df.fillna(df.mean(), inplace=True)
    df.set_index(["Date"], drop=False, inplace=True)
    return df

def predict_with_windows(windows, train, test, lag_features, is_advanced=False):
    for window in windows:
        exogs= []
        for feature in lag_features:
            exogs.append(f"{feature}_mean_lag{window}")
            if is_advanced:
                exogs.append(f"{feature}_std_lag{window}")

        model = auto_arima(train.Close, exogenous=train[exogs], trace=True, error_action="ignore",
                           suppress_warnings=True)
        model.fit(train.Close, exogenous=train[exogs])

        forecast = model.predict(n_periods=len(test), exogenous=test[exogs])
        test[f"Forecast_ARIMAX_{window}d_lag"] = forecast
        test[["Close", f"Forecast_ARIMAX_{window}d_lag"]].plot(figsize=(17, 9))
        print(f"MAPE for {window} day lag prediction is {mape(test['Close'], forecast)}")


def remove_lagged_features_with_window_time(window, features_to_remove, df, lag_features):
    for feature in features_to_remove:
        del df[f"{feature}_mean_lag{window}"]
        lag_features.remove(feature)


def train_test_split_continual(df, test_percentage):
    testNum = round(df.shape[0] * test_percentage)
    train = df.iloc[:-testNum]
    test = df.iloc[-testNum:]

    return train, test

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv('../doge_v1.csv', parse_dates=['Date'], date_parser=dateparse)
    df.set_index(["Date"], drop=False, inplace=True)
    df = df.resample('D').ffill()
    df.reset_index(drop=True, inplace=True)
    lag_features = ["High", "Volume", "twitter_followers", "reddit_average_posts_48h",
                    "reddit_average_comments_48h", "total_issues", "dogecoin_monthly", "dogecoin"]
    df.set_index(["Date"], drop=False, inplace=True)
    df.info()
    windows = [3, 7, 14, 21, 30, 60]
    df = create_lagged_value_columns(lag_features, windows, df)
    testNum = round(df.shape[0] * 0.1)
    train = df.iloc[:-testNum]
    test = df.iloc[-testNum:]
    p_range = range(0,10)
    q_range = range(0,10)
    d_range = [0,1]
    prediction_variable ="Close"
    try_different_model_combinations(p_range, q_range, d_range, train, test, prediction_variable,
                                     lag_features, windows)


