import boxScoreScraper
import csv_cleaner
import gameOutcomePredictor
import teamStatsPredictor
import storeCSV
import sys
from pprint import PrettyPrinter



if __name__ == "__main__":
    pp = PrettyPrinter(indent=4)
    #boxScoreScraper.scrape() # gets the players playing today
    #csv_cleaner.clean()
    teamInfo = gameOutcomePredictor.predict_todays_games() #An array of predicted winners
    teamObj = teamStatsPredictor.predict() #An object of predicted statistics per teams


    # iterate over teamObj and add the predicted stats into teamInfo
    for key, value in teamObj.items():
        currentTeamPredictions = value
        teamInfo[key]["predictedAssists"] = value["assists"]
        teamInfo[key]["predictedTurnovers"] = value["turnovers"]


    storeCSV.store(teamInfo) # store results into a database
