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
    :return:
    """


if __name__ == "__main__":
    dataset = load_dataset("datasets/ATL_2015_to_2018.csv")
    # print(dataset.head(5))
    # general_preview(dataset)
    view_basic_plots(dataset)