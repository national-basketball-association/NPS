import boxScoreScraper
import csv_cleaner
import gameOutcomePredictor
import teamStatsPredictor
import storeCSV
import bettingOdds
import sys
from pprint import PrettyPrinter



if __name__ == "__main__":
    pp = PrettyPrinter(indent=4)
    boxScoreScraper.scrape() # gets the players playing today
    csv_cleaner.clean()
    teamInfo = gameOutcomePredictor.predict_todays_games() #An array of predicted winners
    teamObj = teamStatsPredictor.predict() #An object of predicted statistics per teams


    # iterate over teamObj and add the predicted stats into teamInfo
    for key, value in teamObj.items():
        currentTeamPredictions = value
        teamInfo[key]["predictedAssists"] = value["assists"]
        teamInfo[key]["predictedTurnovers"] = value["turnovers"]
        teamInfo[key]["predictedRebounds"] = value["rebounds"]
        teamInfo[key]["predictedBlocks"] = value["blocks"]
        teamInfo[key]["predictedSteals"] = value["steals"]
        teamInfo[key]["predictedFouls"] = value["fouls"]
        teamInfo[key]["predictedThreePtPercentage"] = value["three_point_percentage"]
        teamInfo[key]["predictedFreeThrowPercentage"] = value["ft_percentage"]


    storeCSV.store(teamInfo) # store results into a database
    bettingOdds.getOdds() # gets betting odds and stores them in db
