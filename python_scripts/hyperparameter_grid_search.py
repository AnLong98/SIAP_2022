import pandas as pd

from python_scripts.sarima_helper import train_test_split_continual
from python_scripts.time_series_utilities import grid_search_hyperparams

if __name__ == '__main__':
    #Load datasets
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df_idena = pd.read_csv('../idena_v1.csv', parse_dates=['Date'], date_parser=dateparse)
    df_idena.set_index(["Date"], drop=False, inplace=True)

    df_doge = pd.read_csv('../doge_v1.csv', parse_dates=['Date'], date_parser=dateparse)
    df_doge.set_index(["Date"], drop=False, inplace=True)

    df_shiba = pd.read_csv('../shiba_v1.csv', parse_dates=['Date'], date_parser=dateparse)
    df_shiba.set_index(["Date"], drop=False, inplace=True)


    df_shiba.drop(df_shiba[df_shiba.Close == 0].index, inplace=True)
    print("------Testing SHIBA INU hyperparameters-----")
    train, test = train_test_split_continual(df_shiba, 0.1)
    #best = grid_search_hyperparams(range(0, 10), range(0, 10), [1], train, test)
    #with open('shiba_grid_result.txt', 'w') as f:
    best = grid_search_hyperparams(range(0, 10), range(0, 10), [1], train, test, dynamic_ahead=True)
    with open('shiba_grid_result_ahead.txt', 'w') as f:
        f.write(best.__str__())

    print("------Testing IDENA hyperparameters-----")
    train, test = train_test_split_continual(df_idena, 0.3)
    best = grid_search_hyperparams(range(0, 10), range(0, 10), [0, 1], train, test, dynamic_ahead=True)
    # best = grid_search_hyperparams(range(0, 10), range(0, 10), [0, 1], train, test)
    # with open('idena_grid_result.txt', 'w') as f:
    with open('idena_grid_result_ahead.txt', 'w') as f:
        f.write(best.__str__())

    print("------Testing DOGE hyperparameters-----")
    train, test = train_test_split_continual(df_doge, 0.1)
    # best = grid_search_hyperparams(range(0, 10), range(0, 10), [1], train, test)
    # with open('doge_grid_result.txt', 'w') as f:
    best = grid_search_hyperparams(range(0, 10), range(0, 10), [1], train, test, dynamic_ahead=True)
    with open('doge_grid_result_ahead.txt', 'w') as f:
        f.write(best.__str__())