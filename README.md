# Introduction

Dogecoin, IDENA and Shiba inu price prediction using ARIMAX and SVR alghoritms. Data was sampled daily from CoinGecko, Yahoo Finance and Google trends. 

ARIMAX model as fitted on 90% of sample data and tested on 10% for Doge and Shiba. IDENA dataset was split in 70:30 ratio. Relevant attributes were taken into account with 3, 7, 14, 21, 30 and 60 day lag to test predictive value. Improvements in model predictivity were made by analyzing correlation heatmaps and introducing average variable value for lag windows along with standard deviation as predictors for one day ahead.
ARIMAX hyperparameters were tuned with Auto Arima stepwise algorithm, criterion for model choice being AIC. Models were evaluated with mean average percent error (MAPE).

More info can be found in my [paper](https://github.com/AnLong98/SIAP_2022/blob/master/izvestaj_pedja_arimax.pdf)

Sample result for IDENA, one day ahead:

![bonitulja](https://github.com/AnLong98/SIAP_2022/blob/master/idena_figures/idena_one_day_ahead_improved.png)

## Technologies

- Python
- Jupyter notebook

## Authors
Predrag Glavas and  Nikola Mijonic

## License
[MIT](https://choosealicense.com/licenses/mit/)

