import itertools
import multiprocessing

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings


def check_stationarity(df, target_var):
    # ADF Test
    result = adfuller(df[target_var].values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

    # KPSS Test
    result = kpss(df[target_var].values, regression='c')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

def plot_pacf_acf(df, target_var):
    plot_acf(df[target_var], lags=50)
    plot_pacf(df[target_var], lags=50)


def test_arima_one_step(train, test, p, d, q, trace=False, display=True, dynamic_ahead=False):
    warnings.filterwarnings('ignore')
    history = [x for x in np.log(train.Close)]
    predictions = list()
    mapes = list()
    for t in range(len(test)):
        model = sm.tsa.ARIMA(history, order=(p, d, q), )
        fitted_model = model.fit()
        output = fitted_model.forecast()
        yhat = output[0]
        predicted = np.exp(yhat)
        predictions.append(predicted)
        expected = test.Close[t]
        if dynamic_ahead:
            history.append(yhat)
        else:
            history.append(np.log(test.Close[t]))
        mape = np.mean(np.abs((expected - predicted) / expected)) * 100
        if trace:
            print('predicted=%f, expected=%f - MAPE = %f' % (predicted, expected, mape))
        mapes.append(mape)
    if display:
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.plot(test.Date, test.Close, 'blue', label='Test data')
        plt.plot(test.Date, predictions, 'green', label='Predicted')
        plt.legend()
        plt.show()
        print('Test MAPE: %.3f' % (np.mean(np.abs((test.Close - predictions) / test.Close)) * 100))
    return predictions, (np.mean(np.abs((test.Close - predictions) / test.Close)) * 100), fitted_model


def test_ARIMA_model_combinations(combinations, train, test,  proc_id, dynamic_ahead):
    models = []
    for combination in combinations:
        print(f"{proc_id}----testing model {combination[0]}, {combination[1]}, {combination[2]} ")

        p = combination[0]
        d = combination[1]
        q = combination[2]

        try:
            _ , mape, model = test_arima_one_step(train, test, p,d,q, display=False, dynamic_ahead=dynamic_ahead)
            models.append(ArimaModelStats(p, d, q, abs(model.aic),
                                               abs(model.bic), mape=mape))

        except:
            print('Model failed with error, resuming..')
    print(f'---process {proc_id} has finished---')
    return models


def grid_search_hyperparams(p_range, q_range, d_range, df_train, df_test, dynamic_ahead=False):
    p_s = p_range
    q_s = q_range
    d_s = d_range

    processor_count = 6

    all_model_combinations = itertools.product(p_s, d_s, q_s)
    all_model_combinations = np.array(list(all_model_combinations))

    process_pool = multiprocessing.Pool(processor_count)
    array = np.array_split(all_model_combinations, processor_count)

    best_models = []
    process_results = []
    for i in range(0, processor_count):
        process_results.append(process_pool.apply_async(test_ARIMA_model_combinations,
                                                        args=(array[i], df_train, df_test, i, dynamic_ahead)))
    for result in process_results:
        result_proc = result.get()
        print(result_proc)
        best_models.extend(result_proc)
        print('Process finished!')

    best_models.sort(key=lambda x: (x.mape, x.aic, x.bic, ))
    best_model = best_models[0]
    best_model.explain_model()
    return best_model



def mape(actual,pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

class ArimaModelStats:
    def __init__(self, p, d, q, aic, bic, mape=0):
        self.p = p
        self.q = q
        self.d = d
        self.aic = aic
        self.bic = bic
        self.mape = mape


    def explain_model(self):
        print(f"ARIMA model ({self.p} {self.d} {self.q}) "
              f"AIC={self.aic} BIC={self.bic}  MAPE={self.mape}\n")

    def __str__(self):
        return (f"ARIMA model ({self.p} {self.d} {self.q}) "
                f"AIC={self.aic} BIC={self.bic}  MAPE={self.mape}\n")
