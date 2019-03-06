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