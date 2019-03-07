import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
import sys
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import datetime
from nba_api.stats.endpoints import scoreboardv2



def load_dataset(filename):
    """
    Given a filename pointing to a CSV file that contains game logs for a
    certain team, returns a pandas dataframe containing the values
    :param filename: CSV file to read in
    :return: Pandas dataframe
    """
    dataset = pandas.read_csv(filename, header=[0])
    return dataset


def create_assists_model(team_abbrev, matchup):
    """
    Given a dataframe
    :param team_abbrev: string representing the team to generate a model that predicts their assists in a game
    :param matchup: the matchup to predict the assists for
    :return:
    """

    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)



    # input features should be matchup, number of assists averaged on the season, win_percentage(?)

    # num assists and win% are in the stats file, so we need to add that to the log dataframe
    log_df["AST_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"] # get the game date for the current game

        tokens = game_date.split("-")
        year = tokens[0] # get the year of the game
        month = tokens[1]

        season = ""

        # determine the formatting of the season by checking whether the month was in the first or second half
        # of the year

        if int(month) >= 6:
            # this is the beginning of a season


            beginning_year = int(year)

            end_year = int(year) + 1

            end_year_str = (str(end_year))[-2:]

            season = "{}-{}".format(str(beginning_year), end_year_str)

        else:
            # this is in the end of a season

            end_year = str(year)

            beginning_year = int(year) - 1

            beginning_year_str = str(beginning_year)

            end_year = end_year[-2:]

            season = "{}-{}".format(beginning_year_str, end_year)

        # the season should be formatted according to the format in the team stats file

        # need to get the team's stats recorded for season

        for stats_index, stats_row in stats_df.iterrows():
            year = stats_df.at[stats_index, "YEAR"]
            if year == season:
                # get the assists and win % from this year
                assists_per_game = stats_df.at[stats_index, "AST"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "AST_SZN_AVG"] = assists_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue


        # assists per game average and win percentage should be in the game log frame now


    print(log_df.head(5))
    sys.exit()










def predictTeamAssists():
    """
    Predicts assists for all teams that are playing in games today
    :return: a dictionary containing the team abbreviation mapped to their predicted assists for their game today
    """

    # call the scoreboard endpoint to get the games happening today
    scoreboard_data = scoreboardv2.ScoreboardV2().get_data_frames()[0]
    time.sleep(2)

    for index, row in scoreboard_data.iterrows():
        # can get the teams playing by getting the GAMECODE of the row
        gamecode = row["GAMECODE"]
        tokens = gamecode.split("/")

        teams_playing_str = tokens[1]

        # slice the string to get the abbreviations of the teams playing
        away_team_abbreviation = teams_playing_str[:3]
        home_team_abbreviation = teams_playing_str[-3:]


        # need to generate an assists model for both of those teams


        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)

        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_assists_model = create_assists_model(away_team_abbreviation, away_matchup)


        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_assists_model = create_assists_model(home_team_abbreviation, home_matchup)


if __name__ == "__main__":
    pandas.set_option('display.max_columns', None)

    predictTeamAssists()