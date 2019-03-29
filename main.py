import boxScoreScraper
import csv_cleaner
import gameOutcomePredictor
import teamStatsPredictor
import storeCSV
import sys
from pprint import PrettyPrinter



if __name__ == "__main__":
    #boxScoreScraper.scrape() # gets the players playing today
    #csv_cleaner.clean()
    predictedWinners = gameOutcomePredictor.predict_todays_games() #An array of predicted winners
    teamObj = teamStatsPredictor.predict() #An object of predicted statistics per teams
    for winner in predictedWinners:
        teamObj[winner]["winPrediction"] = True
    for team in teamObj:
        if "winPrediction" not in teamObj[team]:
            teamObj[team]["winPrediction"] = False
    # pp = PrettyPrinter(indent=4)
    # pp.pprint(teamObj)
    storeCSV.store(teamObj) # store results into a database
