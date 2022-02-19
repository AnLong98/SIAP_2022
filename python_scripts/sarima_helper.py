import itertools
import multiprocessing
from math import sqrt

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ArimaModelStats:
    def __init__(self, p, d, q, P, D, Q, m, aic, bic, rmse, predicted, actual):
        self.p = p
        self.q = q
        self.d = d
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.rmse = rmse
        self.aic = aic
        self.bic = bic
        self.predicted = predicted
        self.actual = actual

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
        print(f"ARIMA model ({self.p} {self.d} {self.q}) ({self.P} {self.D} {self.Q} {self.m}) "
              f"AIC={self.aic} BIC={self.bic} RMSE={self.rmse}\n")

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


def try_different_model_combinations(p_range, q_range, d_range, P_range, D_range, Q_range,
                                     m_range, df_train, df_test, prediction_variable):
    p_s = p_range
    q_s = q_range
    d_s = d_range
    P_s = P_range
    Q_s = Q_range
    D_s = D_range
    m_s = m_range
    best_models = []
    process_results = []
    processor_count = 6

    all_model_combinations = itertools.product(p_s, d_s, q_s, P_s, D_s, Q_s, m_s)
    all_model_combinations = np.array(list(all_model_combinations))

    process_pool = multiprocessing.Pool(processor_count)
    array = np.array_split(all_model_combinations, processor_count)

    for i in range(0, processor_count):
        process_results.append(process_pool.apply_async(test_ARIMA_model_combinations,
                                                        args=(array[i], df_train, prediction_variable, df_test, i)))

    for result in process_results:
        result_proc = result.get()
        print(result_proc)
        best_models.extend(result_proc)
        print('Process finished!')

    best_models.sort(key=lambda x: (x.aic, x.rmse, x.bic))
    best_models = best_models[0:10]
    print("BEST MODELS")
    for model in best_models:
        model.draw_graph()
        model.explain_model()


def test_ARIMA_model_combinations(combinations, df_train, prediction_variable, df_test, proc_id):
    models = []
    for combination in combinations:
        print(f"{proc_id}----testing model {combination[0]}, {combination[1]}, {combination[2]} - {combination[3]},"
              f" {combination[4]}, {combination[5]}, {combination[6]}")

        p = combination[0]
        d = combination[1]
        q = combination[2]
        P = combination[3]
        D = combination[4]
        Q = combination[5]
        m = combination[6]
        try:
            model = ARIMA(df_train[prediction_variable], order=(p, d, q), seasonal_order=(P, D, Q, m))
            model = model.fit()
            start = len(df_train)
            end = len(df_train) + len(df_test) - 1
            pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA predictions')
            rmse = sqrt(mean_squared_error(df_test.Close, pred))
            models.append(ArimaModelStats(p, d, q, P, D, Q, m, abs(model.aic),
                                               abs(model.bic), rmse, pred, df_test.Close))

        except:
            print('Model failed with error, resuming..')
    print(f'---process {proc_id} has finished---')
    return models

if __name__ == '__main__':
    df = pd.read_csv('../doge_v1.csv', index_col='Date', parse_dates=True)
    df = df.resample('w').mean().ffill()
    testNum = round(df.shape[0] * 0.12)
    p_range = range(1, 3)
    q_range = range(25, 30)
    d_range = range(1, 3)
    P_range = [0]
    D_range = [0]
    Q_range = [0]
    m_range = [0]
    train = df.iloc[:-testNum]
    test = df.iloc[-testNum:]
    try_different_model_combinations(p_range, q_range, d_range, P_range, D_range, Q_range,
                                     m_range, train, test, 'Close')