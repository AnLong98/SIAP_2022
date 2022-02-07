from time import sleep

from pycoingecko import CoinGeckoAPI
import datetime
import csv
community_data_headers = ["facebook_likes", "twitter_followers", "reddit_average_posts_48h",
                          "reddit_average_comments_48h", "reddit_subscribers", "reddit_accounts_active_48h"]
developer_data_headers = ["forks", "stars", "subscribers", "total_issues", "closed_issues", "pull_requests_merged",
                          "pull_request_contributors", "commit_count_4_weeks"]


def get_community_data_from_json(json_response):
    community_data = []
    for header in community_data_headers:
        try:
            if not json_response['community_data'] or json_response['community_data'][header] is None:
                community_data.append('')
            else:
                community_data.append(json_response['community_data'][header])
        except:
            community_data.append('')

    return community_data


def get_developer_data_from_json(json_response):
    developer_data = []
    for header in developer_data_headers:
        try:
            if not json_response['developer_data'] or json_response['developer_data'][header] is None:
                developer_data.append('')
            else:
                developer_data.append(json_response['developer_data'][header])
        except:
            developer_data.append('')

    return developer_data


def get_coingecko_data():
    coins = ['shiba-inu','dogecoin', 'idena']
    files = ['shiba-stats.csv','doge-stats.csv', 'idena-stats.csv' ]
    dates = [datetime.date(2020, 8, 1), datetime.date(2017, 11, 9), datetime.date(2020, 8, 11) ]

    for i in range(3):
        # open the file in the write mode
        f = open(files[i], 'w', newline='')

        # create the csv writer
        writer = csv.writer(f)
        header_row = ['date']
        header_row.extend(community_data_headers)
        header_row.extend(developer_data_headers)
        writer.writerow(header_row)
        load_and_store_data(writer, coins[i], dates[i])
        f.close()



def load_and_store_data(writer, coin_name, date_start):
    cg = CoinGeckoAPI()
    rate_limit_minute = 50

    start_date = date_start
    end_date = datetime.date(2022, 2, 4)
    delta = datetime.timedelta(days=1)
    call_cnt = 50
    while start_date <= end_date:
        if call_cnt >= 48:
            sleep(70)
            call_cnt = 0
        row = []
        date_string = f'{start_date:%d-%m-%Y}'
        try:
            stats_for_date = cg.get_coin_history_by_id(id=coin_name, date=date_string, localization=False)
        except:
            #Try again after cooldown
            call_cnt = 50
            continue
        print(call_cnt)
        call_cnt += 1
        row.append(date_string)
        row.extend(get_community_data_from_json(stats_for_date))
        row.extend(get_developer_data_from_json(stats_for_date))

        writer.writerow(row)
        start_date += delta


if __name__ == '__main__':
    get_coingecko_data()
