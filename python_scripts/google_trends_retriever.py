
from pytrends.request import TrendReq
from pytrends import dailydata
search_topics = ["shiba inu coin", 'dogecoin', 'idena']
files = ['shiba_google_trend.csv', 'doge_google_trend.csv', 'idena_google_trend.csv']
start_years = [2020, 2017, 2020]
start_months = [8, 11, 8]
start_days = [1, 9, 11]
start_hours = [0, 0, 0]
end_years = [2022, 2022, 2022]
end_months = [2, 2, 2]

def get_google_trend_data():
    for i in range(3):
        trend = dailydata.get_daily_data(search_topics[i], start_years[i], start_months[i], end_years[i], end_months[i], geo='')
        trend.to_csv(files[i])
        print(trend)


if __name__ == '__main__':
    get_google_trend_data()
