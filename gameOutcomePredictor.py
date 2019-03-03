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
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

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
    :return:
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
        if win_loss == 'W':
            # increment the win streak by 1
            win_streak += 1
            df.at[index, "WIN STREAK"] = win_streak


            # set the WL_BOOL colum to true
            df.at[index, "WL_BOOL"] = 1
        else:
            win_streak = 0
            df.at[index, "WIN STREAK"] = win_streak

            # set WL BOOL column
            df.at[index, "WL_BOOL"] = 0


    # df has some new features


    # need to encode the matchup feature because it is a categorical variable
    le = LabelEncoder()
    matchups = (df["MATCHUP"].values).tolist()
    # print(matchups)
    le.fit(matchups) #fitting the label encoder to the list of different matchups

    # now get a transformation of the matchups column
    matchups_transformed = le.transform(matchups)
    df["MATCHUPS_TRANSFORMED"] = matchups_transformed


    array = df.values

    # print(df.dtypes)

    X = array[:,[28,29, 31,32,34]] # the home team, win_streak, and matchups_transformed features
    # print(len(X))
    Y = array[:,33] # the win loss bool feature
    Y = Y.astype('int')

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)


    scoring = 'accuracy'

    # the test said that decision tree classifier scored well, so we're going with that
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    # predictions = dtc.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))

    # print(le.transform(["ATL vs. CHI"]))


    return dtc


def make_prediction(model, matchup, df):
    # format of the input feature vector should be [NUM_WINS, NUM_LOSSES, HOME_TEAM, WIN_STREAK, TRANSFORMED_MATCHUP)


    le = LabelEncoder()
    # transformed = le.fit_transform((df["MATCHUP"].values).tolist())
    le.fit((df["MATCHUP"].values).tolist())

    # transform the matchup into a number
    transformed_matchup = le.transform([matchup])
    # print(transformed_matchup)


    prediction = model.predict([[15,48,1,1,transformed_matchup]])
    print(prediction)



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


if __name__ == "__main__":
    dataset = load_dataset("datasets/CLE_2015_to_2018.csv")
    # print(dataset.head(5))
    # general_preview(dataset)
    # view_basic_plots(dataset)
    # build_model(dataset)
    model = create_model(dataset)
    # save_model(model, "ATL_Model.sav")
    make_prediction(model, "CLE vs. DET", dataset)

