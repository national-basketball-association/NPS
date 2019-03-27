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


verbose = False
labelEncoder = None

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


    # the log dataframe should be formatted and usable for model training

    # need to encode the matchup feature because it is a categorical variable
    le = LabelEncoder()
    matchups = (log_df["MATCHUP"].values).tolist()
    le.fit(matchups)  # fitting the label encoder to the list of different matchups
    global labelEncoder
    labelEncoder = le

    # now get a transformation of the matchups column
    matchups_transformed = le.transform(matchups)


    log_df["MATCHUPS_TRANSFORMED"] = matchups_transformed

    array = log_df.values

    # now format the input and output feature vectors
    X = array[:, [30, 31, 32]] # this should be the assist season average, the win percentage, and the matchup
    Y = array[:,22] # this should be the assist total for a game
    Y = Y.astype('int')

    # now split into training and testing splits
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    # set the type of scoring
    scoring = 'accuracy'

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    if verbose:
        predictions = dtc.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        print()


    return dtc


def predictTeamAssists():
    """
    Predicts assists for all teams that are playing in games today
    :return: a dictionary containing the team abbreviation mapped to their predicted assists for their game today
    """

    # call the scoreboard endpoint to get the games happening today
    scoreboard_data = scoreboardv2.ScoreboardV2().get_data_frames()[0]
    time.sleep(2)

    predictions = {}

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




        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires assist season average, current winning percentage, and matchup as input variables
        # get the assist season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current assist average and winning percentage
        current_assist_average = 0
        current_winning_percentage = 0
        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_assist_average = away_team_stats.at[team_stats_index, "AST"]
                current_winning_percentage = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found assist average and winning percentage, need to encode the matchup
        # use the label encoder that was fitted earlier
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        print("transformed away matchup is {}".format(transformed_away_matchup))

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_assists_model.predict([
            [
                current_assist_average,
                current_winning_percentage,
                transformed_away_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]


        # now that we have the away team prediction, we can predict the assists for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_assists_model = create_assists_model(home_team_abbreviation, home_matchup)



    return predictions




def create_turnovers_model(team_abbrev, matchup):
    """
    Given a specific team and their matchup, predicts the number of turnovers they will have in that game
    :param team_abbrev:
    :param matchup:
    :return:
    """


if __name__ == "__main__":
    pandas.set_option('display.max_columns', None)


    predictions = predictTeamAssists()
    print(predictions)