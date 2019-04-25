import json
import requests
import csv
from pymongo import MongoClient
import glob
import sys
from pprint import PrettyPrinter

#connect to the database
client = MongoClient("mongodb+srv://rmohamme:green12@cluster0-8eolw.mongodb.net/test?retryWrites=true")
db = client["NPS"]
col = db["BETTING_ODDS"]
col.remove({})
# An api key is emailed to you when you sign up to a plan
api_key = '8b00b7567b859bd2664a484dbb835c95'


# First get a list of in-season sports
sports_response = requests.get('https://api.the-odds-api.com/v3/sports', params={
    'api_key': api_key
})

sports_json = json.loads(sports_response.text)

if not sports_json['success']:
    print(
        'There was a problem with the sports request:',
        sports_json['msg']
    )

else:
    print()
    print(
        'Successfully got {} sports'.format(len(sports_json['data'])),
        'Here\'s the first sport:'
    )
    print(sports_json['data'][0])
    # for x in range(len(print(sports_json['data'])):
    #     print print(sports_json['data'][x]



# To get odds for a sepcific sport, use the sport key from the last request
#   or set sport to "upcoming" to see live and upcoming across all sports
sport_key = 'basketball_nba'
# sport_key ='soccer_spain_la_liga'
odds_response = requests.get('https://api.the-odds-api.com/v3/odds', params={
    'api_key': api_key,
    'sport': sport_key,
    # 'site_key': 'betfair'
    'region': 'us',
    'mkt': 'h2h'
})

odds_json = json.loads(odds_response.text)
if not odds_json['success']:
    print(
        'There was a problem with the odds request:',
        odds_json['msg']
    )

else:
    # odds_json['data'] contains a list of live and
    #   upcoming events and odds for different bookmakers.
    # Events are ordered by start time (live events are first)
    print()
    print(
        'Successfully got {} events'.format(len(odds_json['data'])),
        'Here\'s the first event:'
    )


    # print(odds_json['data'][0])
    # print(*odds_json['data'], sep = "\n")
    for x in odds_json['data']:
        print(x)



        # print(x["teams"])
        # print(x["commence_time"])


        # print(x["sites"][0]["odds"])
        if x["sites_count"] != 0:

            val = round(((x["sites"][0]["odds"]["h2h"][0]) / (x["sites"][0]["odds"]["h2h"][1])) * 100, 0)
            val2 = round((val * - 1) - ((val / 100) * (20)), 0)
            print(val)
            print(val2)
            col.insert({'team1': x["teams"][0], 'team2': x["teams"][1], 'val1': val, 'val2': val2})
        print("\n")
        # col.insert({, 'home': x["home_team"],})
    # Check your usage
    print()
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])
