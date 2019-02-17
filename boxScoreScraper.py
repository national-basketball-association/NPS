import urllib.request, json

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.endpoints import commonallplayers
from nba_api.stats.endpoints import playercareerstats
import time
import sys


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
    # print(filename)
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


def getAllNbaPlayers():
    """
    IMPORTANT: the list contains all players ever, not just current players
    :return: list of dictionaries, each representing an NBA player
    """
    nba_players = players.get_players()
    return nba_players



def getPlayerNameFromId(player_id):
    """
    Given ID of the player, gets the full name of the NBA player associated with the ID
    :param player_id:
    :return:
    """
    nba_players = getAllNbaPlayers()
    curr_player = [player for player in nba_players
                   if player['id'] == player_id][0]

    full_name = curr_player["full_name"]
    print(full_name)
    return full_name



def getPlayerRegularSeasonStats(player_id):
    """
    Given a player id, gets their career regular season stats from the NBA.com API and returns that dataframe
    :param player_id: NBA.com player id (ex. 1495)
    :return: pandas DataFrame containing the stats
    """
    # given a player_id
    stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
    # print(type(stats))
    return stats


def writeRegularSeasonStatsToCsv(player_name, regular_season_stats):
    """

    :param player_name:
    :param regular_season_stats:
    :return:
    """

    # replace spaces in player name with underscore
    player_name.replace(" ", "_")

    # format the filename
    filename = 'datasets/player_stats/{}_Reg_Season_Stats.csv'.format(player_name)
    # print(filename)
    regular_season_stats.to_csv(filename, index=None, header=True)



def commonteamroster(team_id):
    stats = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]

def getAllCurrentPlayerIds():
    """
    Returns a Pandas dataframe containing all current player names and IDs
    :return: Pandas dataframe, use PERSON_ID and DISPLAY_LAST_COMMA_FIRST for the ID and names
    """
    stats = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    return stats

def scrapePlayerStats():
    player_information = getAllCurrentPlayerIds() # get a list of player names and IDs
    for index, row in player_information.iterrows():
        curr_id = row['PERSON_ID']
        curr_full_name = row['DISPLAY_FIRST_LAST']
        formatted_full_name = curr_full_name.replace(" ", "_")


        current_player_stats = playercareerstats.PlayerCareerStats(curr_id).get_data_frames()[0]
        filename = 'datasets/player_stats/{}_Stats.csv'.format(formatted_full_name)

        current_player_stats.to_csv(filename, index=None, header=True)
        print("Wrote to {}".format(filename))
        time.sleep(5)


if __name__ == "__main__":
    # stats = getAllCurrentPlayerIds()
    # filename = 'datasets/test.csv'
    # stats.to_csv(filename)
    scrapePlayerStats()
