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
import numpy
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


def general_preview(dataset):
    """
    Given a pandas dataframe, shows a general description of the data
    :param dataset:
    :return:
    """
    print("Shape: {}".format(dataset.shape))
    print("Head:")
    print(dataset.head(5))
    print("Description:")
    print(dataset.describe())

    # grouping by wins/losses
    print("Grouping by wins/losses:")
    print(dataset.groupby("WL").size())


def view_basic_plots(dataset):
    # create box and whisker plots
    # dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # plt.show()
    scatter_matrix(dataset)
    plt.show()


def create_model(dataset):
    """
    idk what im doing so im making this other method to try new stuff
    :return: A DecisionTreeClassifier model that can predict game outcomes for the team that the dataset belongs to
    """

    pandas.set_option('display.max_columns', None) # setting options to display all columns when printing dataframe

    df = dataset

    df = df.sort_values("GAME_DATE")

    df["HOME WIN"] = 0
    df["HOME TEAM"] = 0


    win_streak = 0

    # iterate over the data to find when this team won on their home court
    for index, row in df.iterrows():
        matchup = df.at[index, "MATCHUP"] # get the matchup for the current game


        # record whether they were the home team
        if "vs." in matchup:
            df.at[index ,"HOME TEAM"] = 1
        else:
            df.at[index, "HOME TEAM"] = 0

        # record whether they won as the home team
        if "vs." in matchup:
            # if they are the home team, check if they won
            win_loss = df.at[index, "WL"]
            # print("WIN LOSS IS  {}".format(win_loss))
            if win_loss == "W":
                # this means they were the home team and won, so record that value
                df.at[index, "HOME WIN"] = 1
            else:
                df.at[index, "HOME WIN"] = 0
        else:
            df.at[index, "HOME WIN"] = 1


        # add a feature that tracks their current winning streak
        win_loss = df.at[index, "WL"] # whether they won the game or not

        # positive numbers represent a winning streak, negative numbers represent a losing streak

        # print(matchup)
        if win_loss == 'W' and win_streak >= 0:
            # either a new win streak and continuing win streak, so increment by 1
            win_streak += 1
            df.at[index, "WIN STREAK"] = win_streak

            df.at[index, "WL_BOOL"] = 1
        elif win_loss == 'W' and win_streak < 0:
            # they were on a losing streak, this breaks it
            win_streak = 1
            df.at[index, "WIN STREAK"] = win_streak

            df.at[index, "WL_BOOL"] = 1
        elif win_loss == 'L' and win_streak > 0:
            # they were on a winning streak, this breaks it, so set to -1
            win_streak = -1
            df.at[index, "WIN STREAK"] = win_streak

            df.at[index, "WL_BOOL"] = 0
        elif win_loss == 'L' and win_streak <= 0:
            # they were on a losing streak, so decrement by 1
            win_streak -= 1
            df.at[index, "WIN STREAK"] = win_streak

            df.at[index, "WL_BOOL"] = 0

        # print(win_streak)


    # df has some new features


    # need to encode the matchup feature because it is a categorical variable
    le = LabelEncoder()
    matchups = (df["MATCHUP"].values).tolist()
    le.fit(matchups) #fitting the label encoder to the list of different matchups

    # now get a transformation of the matchups column
    matchups_transformed = le.transform(matchups)
    df["MATCHUPS_TRANSFORMED"] = matchups_transformed


    array = df.values


    # print(df.head(5))
    # print(df.tail(2))
    # print(df[df['WL_BOOL'].isnull()])
    # sys.exit(1)
    X = array[:,[28,29, 31,32,34]] # the home team, win_streak, and matchups_transformed features
    Y = array[:,33] # the win loss bool feature
    # Y = array[:,27] # testing to see how it works with predicting point spread
    Y = Y.astype('int')

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)


    scoring = 'accuracy'

    # the test said that decision tree classifier scored well, so we're going with that
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    predictions = dtc.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))
    # print()

    # print(le.transform(["ATL vs. CHI"]))


    return dtc


def make_prediction(model, matchup, df):
    """
    Makes a prediction using the given ml model, the given NBA matchup, and a dataframe containing the teams data
    :param model:
    :param matchup:
    :param df:
    :return:
    """
    # format of the input feature vector should be [NUM_WINS, NUM_LOSSES, HOME_TEAM, WIN_STREAK, TRANSFORMED_MATCHUP)

    tokens = matchup.split(" ")
    team = tokens[0]

    wins_losses = get_team_record(team) # get the current record of the team

    # check whether this is a home game for the team the model was trained for
    is_home = 1

    if "@" in tokens[1]:
        is_home = 0

    le = LabelEncoder()
    # transformed = le.fit_transform((df["MATCHUP"].values).tolist())
    le.fit((df["MATCHUP"].values).tolist())

    # transform the matchup into a number
    transformed_matchup = le.transform([matchup])
    # print(transformed_matchup)



    # get the team's current win_streak
    win_streak = get_team_winstreak(df)
    prediction = model.predict([
        [wins_losses[0],
         wins_losses[1],
         is_home,
         win_streak,
         transformed_matchup]])
    #
    # if 1 in prediction:
    #     print("{} will win!".format(team))
    # else:
    #     print("{} will lose!".format(team))
    #

    return prediction


def get_team_winstreak(df):
    """
    iterates over the rows in the given dataframe and calculates the teams current winning streak
    :param df:
    :return:
    """
    df = df[::-1]

    # iterate over the rows and count the current win streak
    win_streak = 0
    won_last_game = 2
    for index, row in df.iterrows():
        # win_or_loss = df.at[index, "WL"]
        # if win_or_loss == 'W':
        #     win_streak += 1
        # else:
        #     # the loss breaks the streak
        #     # return win_streak
        #     return win_streak
        win_or_loss = df.at[index, "WL"]
        if win_or_loss == 'W':
            # this means they're on a winning streak
            if win_streak == 0:
                # their most recent game was a W
                win_streak += 1
            else:
                if win_streak > 0:
                    # still counting wins
                    win_streak += 1
                else:
                    return win_streak

        else:
            # this means they're on a losing streak
            if win_streak == 0:
                # their most recent game was a L
                win_streak -= 1
            else:
                # not looking at the most recent game
                if win_streak > 0:
                    # this is a win, so it breaks the streak of losses, return win_streak
                    return win_streak
                else:
                    win_streak -= 1


def get_team_record(team_abbrev):
    """
    Given the abbreviation for a team, returns their current record
    :param team_abbrev:
    :return:
    """
    filename = "datasets/team_stats/" + team_abbrev + "_Stats_By_Year.csv"

    # print(filename)

    df = load_dataset(filename) # load the data containing team stats

    # print(df.iloc[-1:])

    last_row = df.iloc[-1:]

    wins = (last_row["WINS"].values)[0]

    losses = (last_row["LOSSES"].values)[0]

    wins_losses = []

    wins_losses.append(wins)
    wins_losses.append(losses)

    return wins_losses


def save_model(model, filename):
    """
    Given an ML model and a filename, saves the model to that file so it can be reloaded later
    :param model:
    :param filename:
    :return:
    """
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def test_models(X_train, Y_train, scoring, seed):
    """
    Given sets of data, tests which model is best to use
    :param X_train:
    :param Y_train:
    :param scoring:
    :param seed:
    :return:
    """
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def calculate_home_win_percentage(df):
    """
    Given a dataset created by predicting(), calculates the home win percentage. That number can serve as a baseline
    to test the effectiveness of the mdoel
    :param df:
    :return:
    """
    num_home_wins = df["HOME WIN"].sum()
    num_home_games = df["HOME TEAM"].sum()
    win_percentage = num_home_wins / num_home_games

    print('Home Win percentage: {0:.2f}%'.format(100 * win_percentage))


def predict_todays_games():
    """
    Creates a model for all the games happening today and tries to predict the outcomes
    :return:
    """

    # call the scoreboard endpoint to get the games happening today
    scoreboard_data = scoreboardv2.ScoreboardV2().get_data_frames()[0]
    time.sleep(2)

    winners = []


    for index, row in scoreboard_data.iterrows():

        # can get the teams playing by getting the GAMECODE of the row
        gamecode = row["GAMECODE"]
        tokens = gamecode.split("/")

        teams_playing_str = tokens[1]

        # slice the string to get the abbreviations of the teams playing
        away_team_abbreviation = teams_playing_str[:3]
        home_team_abbreviation = teams_playing_str[-3:]

        # format a matchup string using the abbreviations
        matchup = "{} @ {}".format(away_team_abbreviation, home_team_abbreviation)

        # get the dataframe for the away team
        filename = "datasets/{}_2015_to_2018.csv".format(away_team_abbreviation)
        df = load_dataset(filename) # load a dataframe for the teams data

        # create a model for the current team
        model = create_model(df)

        prediction = make_prediction(model, matchup, df)

        # print("The predicted point spread for {} was {}".format(away_team_abbreviation, prediction[0]))

        if 1 in prediction:
            winners.append(away_team_abbreviation)
        else:
            winners.append(home_team_abbreviation)


    # should have predicted a winner for all the games, go through and print all the winners to the console
    for x in winners:
        print("I think {} will win!".format(x))

    return winners


if __name__ == "__main__":
    predict_todays_games()

