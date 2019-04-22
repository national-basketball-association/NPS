import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
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
import pprint


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


def create_assists_model(team_abbrev):
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

        away_assists_model = create_assists_model(away_team_abbreviation)




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

        home_assists_model = create_assists_model(home_team_abbreviation)

        # now make a prediction for the home team
        # the model requires assist season average, current winning percentage, and matchup as input variables
        # get the assist season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current assist average and winning percentage
        current_assist_average = 0
        current_winning_percentage = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_assist_average = home_team_stats.at[team_stats_index, "AST"]
                current_winning_percentage = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found assist average and winning percentage, need to encode the matchup
        # use the label encoder that was fitted earlier
        # global labelEncoder
        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_assists_model.predict([
            [
                current_assist_average,
                current_winning_percentage,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]


    return predictions




def create_turnovers_model(team_abbrev, matchup):
    """
    Given a specific team and their matchup, predicts the number of turnovers they will have in that game
    :param team_abbrev: a string referring to the team the model is being trained for
    :param matchup: the matchup to predict the turnovers for
    :return: a model that predicts the number of turnovers a team will have in their upcoming matchup
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # num turnovers and win% are in the stats file, so we need to add that to the log dataframe
    log_df["TOV_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0


    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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
                # get the turnovers and win % from this year
                turnovers_per_game = stats_df.at[stats_index, "TOV"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "TOV_SZN_AVG"] = turnovers_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue


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

    X = array[:, [30, 31, 32]]  # this should be the turnover season average, the win percentage, and the matchup
    Y = array[:, 25]  # this should be the turnover total for a game
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


def predictTeamTurnovers():
    """
    Predicts turnovers for all the teams that are playing in games today
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

        # need to generate an turnovers model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_turnovers_model = create_turnovers_model(away_team_abbreviation, away_matchup)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires turnovers season average, current winning percentage, and matchup as input variables
        # get the turnovers season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current turnover average and winning percentage
        current_turnover_average = 0
        current_winning_percentage = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_turnover_average = away_team_stats.at[team_stats_index, "TOV"]
                current_winning_percentage = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break


        # found the turnover average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_turnovers_model.predict([
            [
                current_turnover_average,
                current_winning_percentage,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the turnovers for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_turnovers_model = create_turnovers_model(home_team_abbreviation, home_matchup)

        # now make a prediction for the home team
        # the model requires turnovers season average, current winning percentage, and matchup as input variables
        # get the turnovers season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current turnovers average and winning percentage
        current_turnover_average = 0
        current_winning_percentage = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_turnover_average = home_team_stats.at[team_stats_index, "TOV"]
                current_winning_percentage = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the turnover average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_turnovers_model.predict([
            [
                current_turnover_average,
                current_winning_percentage,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]


    return predictions


def create_rebound_model(team_abbrev, matchup):
    """
    Given a specific NBA team and their matchup in a game, creates a model
    :param team_abbrev: a string referring to the team the model is being trained for
    :param matchup: the matchup to predict the turnovers for
    :return: a model that predicts the number of rebounds a team will have in their upcoming matchup
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # num rebound and win% are in the stats file, so we need to add that to the log dataframe
    log_df["REB_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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
                # get the rebounds and win % from this year
                reb_per_game = stats_df.at[stats_index, "REB"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "REB_SZN_AVG"] = reb_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue


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
    X = array[:, [30, 31, 32]] # this is the rebound season average, win percentage, and matchup
    Y = array[:, 21]  # this should be the rebound total for a game
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

def predictTeamRebounds():
    """
    Predicts team rebounds for all the teams playing in games happening today
    :return: a dictionary mapping the abbreviation of the team to the number of rebounds NPS predicts for them
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

        # need to generate an rebounds model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_rebounds_model = create_rebound_model(away_team_abbreviation, away_matchup)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires rebounds season average, current winning percentage, and matchup as input variables
        # get the rebounds season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current rebound average and winning percentage
        current_rebound_average = 0
        current_winning_percentage = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_rebound_average = away_team_stats.at[team_stats_index, "REB"]
                current_winning_percentage = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the rebound average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_rebounds_model.predict([
            [
                current_rebound_average,
                current_winning_percentage,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the rebounds for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_rebound_model = create_rebound_model(home_team_abbreviation, home_matchup)

        # now make a prediction for the home team
        # the model requires rebound season average, current winning percentage, and matchup as input variables
        # get the rebound season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current rebound average and winning percentage
        current_rebound_average = 0
        current_winning_percentage = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_rebound_average = home_team_stats.at[team_stats_index, "REB"]
                current_winning_percentage = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the rebound average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_rebound_model.predict([
            [
                current_rebound_average,
                current_winning_percentage,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]

    # return the dictionary of team abbreviations and their rebound predictions
    return predictions


def create_blocks_model(team_abbrev):
    """
    Given an NBA team and their upcoming matchup, predicts the number of blocks the team will have in that game
    Uses win percentage, matchup, and average blocks in the current season as input variables
    Outputs a prediction for blocks in a game
    :param team_abbrev: 3 letter abbreviation for NBA team, ex. BOS
    :return: a DecisionTreeClassifier that can be used to predict blocks for the given team in their matchup
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # num blocks and win% are in the stats file, so we need to add that to the log dataframe
    log_df["BLK_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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
                # get the blocks and win % from this year
                blk_per_game = stats_df.at[stats_index, "BLK"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "BLK_SZN_AVG"] = blk_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue

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
    X = array[:, [30, 31, 32]]  # this is the block season average, win percentage, and matchup
    Y = array[:, 24]  # this should be the block total for a game
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

def predict_team_blocks():
    """
    Predicts block numbers for all the teams playing in games on the current day
    :return: a dictionary mapping the team names to their predicted block numbers
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

        # need to generate an blocks model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_blocks_model = create_blocks_model(away_team_abbreviation)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires blocks season average, current winning percentage, and matchup as input variables
        # get the blocks season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current block average and winning percentage
        current_block_average = 0
        current_winning_percentage = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_block_average = away_team_stats.at[team_stats_index, "BLK"]
                current_winning_percentage = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the block average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_blocks_model.predict([
            [
                current_block_average,
                current_winning_percentage,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the blocks for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_block_model = create_blocks_model(home_team_abbreviation)

        # now make a prediction for the home team
        # the model requires block season average, current winning percentage, and matchup as input variables
        # get the block season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current block average and winning percentage
        current_block_average = 0
        current_winning_percentage = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_block_average = home_team_stats.at[team_stats_index, "BLK"]
                current_winning_percentage = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the block average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_block_model.predict([
            [
                current_block_average,
                current_winning_percentage,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]

    # return the dictionary of team abbreviations and their rebound predictions
    return predictions

def create_steals_model(team_abbrev):
    """
    Creates a model for a given NBA team that can be used to predict their steals in a particular matchup
    :param team_abbrev: three letter abbreviation for NBA team, ex. BOS
    :return: type <DecisionTreeClassifier> that can be used to predict steal outcomes for a single game
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # num steals and win% are in the stats file, so we need to add that to the log dataframe
    log_df["STL_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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
                # get the steals and win % from this year
                stl_per_game = stats_df.at[stats_index, "STL"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "STL_SZN_AVG"] = stl_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue

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
    X = array[:, [30, 31, 32]]  # this is the steal season average, win percentage, and matchup
    Y = array[:, 23]  # this should be the steal total for a game
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


def predict_team_steals():
    """
    Predicts steals numbers for all the teams playing in games today
    :return: a dictionary mapping team names to the number of steals they are predicted to achieve in their upcoming
    game
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

        # need to generate an steals model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_steals_model = create_steals_model(away_team_abbreviation)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires steals season average, current winning percentage, and matchup as input variables
        # get the steals season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current steal average and winning percentage
        current_steal_average = 0
        current_winning_pct = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_steal_average = away_team_stats.at[team_stats_index, "STL"]

                current_winning_pct = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the steal average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_steals_model.predict([
            [
                current_steal_average,
                current_winning_pct,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the steals for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_steals_model = create_steals_model(home_team_abbreviation)

        # now make a prediction for the home team
        # the model requires steal season average, current winning percentage, and matchup as input variables
        # get the steal season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current steal average and winning percentage
        current_steal_average = 0
        current_winning_pct = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_steal_average = home_team_stats.at[team_stats_index, "STL"]
                current_winning_pct = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the rebound average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_steals_model.predict([
            [
                current_steal_average,
                current_winning_pct,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]

    # return the dictionary of team abbreviations and their steal predictions
    return predictions


def create_fouls_model(team_abbrev):
    """
    Given a team abbreviation referring to an NBA team, creates a DTC model that can be used to predict the number
    of fouls that team will accrue in a given matchup
    :param team_abbrev: 3 letter abbreviation for an NBA team, such as BOS or ATL
    :return: <DecisionTreeClassifier> model that can be used to predict fouls for the given team
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # average fouls and win% are in the stats file, so we need to add that to the log dataframe
    log_df["PF_SZN_AVG"] = 0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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
                # get the fouls and win % from this year
                pf_per_game = stats_df.at[stats_index, "PF"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "PF_SZN_AVG"] = pf_per_game
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue

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
    X = array[:, [30, 31, 32]]  # this is the pf season average, win percentage, and matchup
    Y = array[:, 26]  # this should be the pf total for a game
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


def predict_team_fouls():
    """
    Predicts foul totals for all the teams that are playing today
    :return: a dictionary mapping the team abbreviation to the number of fouls they are predicted to get today
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

        # need to generate a fouls model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_fouls_model = create_fouls_model(away_team_abbreviation)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires fouls season average, current winning percentage, and matchup as input variables
        # get the fouls season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current fouls average and winning percentage
        current_fouls_average = 0
        current_winning_pct = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_fouls_average = away_team_stats.at[team_stats_index, "PF"]

                current_winning_pct = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the fouls average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_fouls_model.predict([
            [
                current_fouls_average,
                current_winning_pct,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the assists for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_fouls_model = create_fouls_model(home_team_abbreviation)

        # now make a prediction for the home team
        # the model requires pf season average, current winning percentage, and matchup as input variables
        # get the pf season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current foul average and winning percentage
        current_fouls_average = 0
        current_winning_pct = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_fouls_average = home_team_stats.at[team_stats_index, "PF"]
                current_winning_pct = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the pf average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_fouls_model.predict([
            [
                current_fouls_average,
                current_winning_pct,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]

    # return the dictionary of team abbreviations and their rebound predictions
    return predictions



def create_three_point_model(team_abbrev):
    """
    Given an NBA team, creates a model that can be used to predict their three point percentage against another NBA
    team
    :param team_abbrev: 3 letter abbreviation used to refer to an NBA team, such as BOS or ATL
    :return:
    """
    # first need to load the game logs
    log_filename = "datasets/{}_2015_to_2018.csv".format(team_abbrev)
    log_df = load_dataset(log_filename)

    # now load the team stats
    stats_filename = "datasets/team_stats/{}_Stats_By_Year.csv".format(team_abbrev)
    stats_df = load_dataset(stats_filename)

    # average 3pt% and win% are in the stats file, so we need to add that to the log dataframe
    log_df["3PT_SZN_AVG"] = 0.0
    log_df["WIN_PCT"] = 0.0

    for index, row in log_df.iterrows():
        game_date = log_df.at[index, "GAME_DATE"]

        tokens = game_date.split("-")
        year = tokens[0]
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

        three_point_average = 0.0

        for stats_index, stats_row in stats_df.iterrows():
            year = stats_df.at[stats_index, "YEAR"]
            if year == season:
                # get the 3pt% and win % from this year
                three_point_average = stats_df.at[stats_index, "FG3_PCT"]
                win_pct = stats_df.at[stats_index, "WIN_PCT"]

                # got the values needed from this season, now add them to the game log dataframe
                log_df.at[index, "3PT_SZN_AVG"] = float(three_point_average)
                log_df.at[index, "WIN_PCT"] = win_pct

                break
            else:
                continue

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
    X = array[:, [30, 31, 32]]  # this is the 3pt% season average, win percentage, and matchup
    Y = array[:, 15]  # this should be the 3pt% total for a game
    Y = Y.astype('float')

    # now split into training and testing splits
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    # set the type of scoring
    scoring = 'accuracy'

    clf = LinearRegression() # have to use a linear regression algorithm in order to predict floats
    clf.fit(X_train, Y_train)
    if verbose:
        predictions = clf.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        print()

    return clf

def predict_team_three_pt_percentage():
    """
    Predicts the three point percentage for all the teams playing today
    :return: dictionary mapping team abbreviations to their predicted 3pt %
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

        # need to generate a 3pt% model for both of those teams
        # format a matchup string using the abbreviations
        away_matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)
        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename)  # load a dataframe for the teams data

        away_three_pt_model = create_three_point_model(away_team_abbreviation)

        # we now have a model for both the home and away team in the current matchup
        # use the model to make a prediction
        # first make a prediction for the away team
        # the model requires 3pt% season average, current winning percentage, and matchup as input variables
        # get the fouls season average
        away_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current fouls average and winning percentage
        current_three_pt_average = 0
        current_winning_pct = 0

        for team_stats_index, team_stats_row in away_team_stats.iterrows():
            year = away_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_three_pt_average = away_team_stats.at[team_stats_index, "FG3_PCT"]

                current_winning_pct = away_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the fouls average and winning percentage
        # encode the matchup using the global labelEncoder
        global labelEncoder
        le = labelEncoder
        transformed_away_matchup = le.transform(["{} @ {}".format(away_team_abbreviation, home_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        away_team_prediction = away_three_pt_model.predict([
            [
                current_three_pt_average,
                current_winning_pct,
                transformed_away_matchup
            ]
        ])
        # store this prediction in the dictionary that will be returned
        predictions[away_team_abbreviation] = away_team_prediction[0]

        # now that we have the away team prediction, we can predict the 3pt% for the home team
        home_matchup = "{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)

        home_three_pt_model = create_three_point_model(home_team_abbreviation)

        # now make a prediction for the home team
        # the model requires pf season average, current winning percentage, and matchup as input variables
        # get the pf season average
        home_team_stats = load_dataset("datasets/team_stats/{}_Stats_By_Year.csv".format(away_team_abbreviation))

        # iterate over the team stats, find their current foul average and winning percentage
        current_three_pt_average = 0
        current_winning_pct = 0
        for team_stats_index, team_stats_row in home_team_stats.iterrows():
            year = home_team_stats.at[team_stats_index, "YEAR"]
            if year == "2018-19":
                # found the current year
                current_three_pt_average = home_team_stats.at[team_stats_index, "FG3_PCT"]
                current_winning_pct = home_team_stats.at[team_stats_index, "WIN_PCT"]
                break

        # found the pf average and winning percentage, need to encode the matchup
        # use the global label encoder

        le = labelEncoder
        transformed_home_matchup = le.transform(["{} vs. {}".format(home_team_abbreviation, away_team_abbreviation)])

        # stored all the inputs, can make a prediction now
        home_team_prediction = home_three_pt_model.predict([
            [
                current_three_pt_average,
                current_winning_pct,
                transformed_home_matchup
            ]
        ])

        # store this prediction in the dictionary that will be returned
        predictions[home_team_abbreviation] = home_team_prediction[0]

    # return the dictionary of team abbreviations and their rebound predictions
    return predictions

def predict():
    """
    Calls helper methods to make predictions for all the team statistics that are needed
    :return: a dictionary mapping teams to the stats they are predicted to achieve in their next matchup
    """
    pandas.set_option('display.max_columns', None)
    predictions = predictTeamAssists()

    teamObj = {}
    for team in predictions:
        teamObj[team] = {"assists": str(predictions[team])}

    # add the turnover predictions
    turnovers = predictTeamTurnovers()
    for team in turnovers:
        teamObj[team]["turnovers"] = str(turnovers[team])

    # add the rebound predictions
    rebounds = predictTeamRebounds()
    for team in rebounds:
        teamObj[team]["rebounds"] = str(rebounds[team])

    # add the block predictions
    blocks = predict_team_blocks()
    for team in blocks:
        teamObj[team]["blocks"] = str(blocks[team])

    # add the steals predictions
    steals = predict_team_steals()
    for team in steals:
        teamObj[team]["steals"] = str(steals[team])

    # add the fouls predictions
    fouls = predict_team_fouls()
    for team in fouls:
        teamObj[team]["fouls"] = str(fouls[team])

    # add the 3pt% predictions
    three_pt_percentage = predict_team_three_pt_percentage()
    for team in three_pt_percentage:
        value = three_pt_percentage[team]
        value = round(value, 3)
        teamObj[team]["three_point_percentage"] = str(value)

    return teamObj

if __name__ == "__main__":
    pandas.set_option('display.max_columns', None)
    # create_steals_model("BOS")
    # create_fouls_model("BOS")