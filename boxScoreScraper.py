import urllib.request, json

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
import time


teamToIndex = {
    'ATL': 0,
    'BOS': 1,
    'BKN': 2,
    'CHA': 3,
    'CHI': 4,
    'CLE': 5,
    'DAL': 6,
    'DEN': 7,
    'DET': 8,
    'GSW': 9,
    'HOU': 10,
    'IND': 11,
    'LAC': 12,
    'LAL': 13,
    'MEM': 14,
    'MIA': 15,
    'MIL': 16,
    'MIN': 17,
    'NOP': 18,
    'NYK': 19,
    'OKC': 20,
    'ORL': 21,
    'PHI': 22,
    'PHX': 23,
    'POR': 24,
    'SAC': 25,
    'SAS': 26,
    'TOR': 27,
    'UTA': 28,
    'WAS': 29,
}

# loads box scores for a team into data frame
def getTeamBoxScoreForYear(teamName, season):
    teamNameId = getTeamIdFromName(teamName)
    # this should find all games where celtics were playing
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=teamNameId)
    games = gamefinder.get_data_frames()[0]

    #filter to the season required
    games_in_season = games[games.SEASON_ID.str[-4:] == season[:4]]
    return games_in_season



# gets all the box scores for a team between the given years and writes the data frame to a csv file
def getTeamBoxScoresBetweenYears(teamName, start_year, end_year):
    frame = getTeamBoxScoreForYear(teamName, formatYearToSeason(start_year))
    for x in range(end_year-start_year):
        season = formatYearToSeason(start_year+1+x)
        frame = frame.append(getTeamBoxScoreForYear(teamName, season), ignore_index=True)


    filename = 'datasets/{}_{}_to_{}.csv'.format(teamName, start_year, end_year)
    print(filename)
    frame.to_csv(filename, index=None, header=True)

    return frame


# takes a year as input, ex. 2017, and formats it to an NBA season, ex 2017-18
def formatYearToSeason(year):
    season = str(year) + "-" + str(year+1)[2:] # needs to be in the format 2017-18
    return season


# given a team abbreviation, gets the ID for the team
def getTeamIdFromName(teamName):
    nba_teams = teams.get_teams()
    team_dict = [team for team in nba_teams
         if team['abbreviation'] == teamName][0]
    return team_dict['id']



# Gets all box scores for every NBA team between the provided years
def getAllTeamBoxScoresBetweenYears(start_year, end_year):
    # iterate over all the team name abbreviations
    for key in teamToIndex.keys():
        getTeamBoxScoresBetweenYears(key, start_year, end_year) # call the helper method with the current team
        time.sleep(10) # without this line, the API sends a connection timeout error after the first couple requests



if __name__ == "__main__":
    # frame = getTeamBoxScoresBetweenYears('MEM', 2015, 2018)
    getAllTeamBoxScoresBetweenYears(2015, 2018)
