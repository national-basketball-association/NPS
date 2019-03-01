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



def build_model(dataset):
    """
    Given a dataset containing information about a team's games, this will build and return a ML model
    :param dataset: pandas dataframe containing game logs for one team
    :return: ML model
    """


    array = dataset.values
    # print(type(array))
    # print(array[:,6]

    X = array[:,6] # this is the matchup, which would be the input if we were making a prediction

    Y = array[:, [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]] # this is the output
    # print(Y)


    validation_size = 0.20

    seed = 14

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)


    # the data has been split into training and testing splits

    


    # models = []
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC(gamma='auto')))


if __name__ == "__main__":
    dataset = load_dataset("datasets/ATL_2015_to_2018.csv")
    # print(dataset.head(5))
    # general_preview(dataset)
    # view_basic_plots(dataset)
    build_model(dataset)
