import urllib.request, json

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.endpoints import commonallplayers
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import scoreboardv2
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
        print("finished {}".format(key))
        time.sleep(2) # without this line, the API sends a connection timeout error after the first couple requests


def getAllNbaPlayers():
    """
    IMPORTANT: the list contains all players ever, not just current players
    :return: list of dictionaries, each representing an NBA player
    """
    nba_players = players.get_players()
    return nba_players


def getAllNbaTeams():
    """
    IMPORTANT: the list contains the teams in the NBA
    :return:
    """
    nba_teams = teams.get_teams()
    return nba_teams



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



# def commonteamroster(team_id):
#     stats = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]

def getAllCurrentPlayerIds():
    """
    Returns a Pandas dataframe containing all current player names and IDs
    :return: Pandas dataframe, use PERSON_ID and DISPLAY_LAST_COMMA_FIRST for the ID and names
    """
    stats = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    return stats

def getTodaysPlayers():

    """
    Returns a list of player IDs belonging to players that played today
    :return:
    """
    data = scoreboardv2.ScoreboardV2().get_data_frames()[0]

    player_ids = []

    # print(data.to_string())
    # print(list(data))
    playing_team_ids = []
    home_team_ids = data["HOME_TEAM_ID"].tolist()
    print(home_team_ids)
    playing_team_ids += home_team_ids

    print(data.to_string())

    away_team_ids = data["VISITOR_TEAM_ID"].tolist()

    playing_team_ids += away_team_ids



    for team_id in playing_team_ids:
        # curr_team = [team for team in getAllNbaTeams()
        #              if team['id'] == team_id]

        current_team_roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        time.sleep(5)
        for index, row in current_team_roster.iterrows():
            player_id = row['PLAYER_ID']
            player_ids.append(player_id)



    print(player_ids)
    return player_ids






def scrapePlayerStats():
    player_information = getAllCurrentPlayerIds() # get a list of player names and IDs
    for index, row in player_information.iterrows():
        curr_id = row['PERSON_ID']
        curr_full_name = row['DISPLAY_FIRST_LAST']
        formatted_full_name = curr_full_name.replace(" ", "_")


        current_player_stats = playercareerstats.PlayerCareerStats(curr_id).get_data_frames()[0]
        print(current_player_stats)
        filename = 'datasets/player_stats/{}_Stats.csv'.format(formatted_full_name)

        current_player_stats.to_csv(filename, index=None, header=True)
        print("Wrote to {}".format(filename))
        time.sleep(5)

def scrapeTodaysPlayerStats(player_ids):
    for player_id in player_ids:
        player_info = [player for player in getAllNbaPlayers()
                       if player['id'] == player_id]
        # print(player_info)
        if player_info == []:
            continue
        player_dict = player_info[0]

        player_name = player_dict['full_name']
        # print(player_name)
        # sys.exit(1)
        formatted_name = player_name.replace(" ", "_")
        current_player_stats = playercareerstats.PlayerCareerStats(player_id).get_data_frames()[0]
        print(current_player_stats)
        filename = 'datasets/player_stats/{}_Stats.csv'.format(formatted_name)

        current_player_stats.to_csv(filename, index=None, header=True)
        print("Wrote to {}".format(filename))
        time.sleep(5)


def scrapeTeamRosters():
    teams = getAllNbaTeams()
    # print(teams)
    for team in teams:
        # print(team)
        team_id = team['id']
        team_abbrev = team['abbreviation']

        time.sleep(5)
        current_team_roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        print(current_team_roster)
        filename = 'datasets/rosters/{}_Roster.csv'.format(team_abbrev)
        current_team_roster.to_csv(filename, index=None, header=True)
        print("finished {}'s roster".format(team_abbrev))



if __name__ == "__main__":
    getAllTeamBoxScoresBetweenYears(2015, 2018)
    scrapePlayerStats()
    scrapeTeamRosters()
    getTodaysPlayers()
    issaList = [201567, 203917, 1629012, 1627737, 1628417, 2747, 203903, 101112, 1628021, 202684, 1626224, 203521, 202688, 1626204, 203089, 1629061, 1627790, 1628383, 203118, 101161, 1629312, 1629015, 203516, 204456, 203613, 200755, 1629003, 203954, 202710, 1627732, 1627788, 202699, 1628413, 1626246, 1629066, 1626156, 201162, 203915, 201960, 1629033, 203925, 203894, 1626210, 202334, 1626203, 1627747, 1626178, 1629058, 1628386, 203459, 1628979, 201588, 1628391, 202339, 1628978, 201572, 1627763, 203503, 203114, 1628425, 1626192, 1628537, 203507, 1626174, 202703, 1629045, 101141, 202326, 1627814, 1628395, 1626188, 1626172, 2738, 1628980, 202691, 1627745, 201973, 203110, 1628035, 201939, 1629094, 2733, 201142, 1628398, 203484, 1628366, 1628404, 1627936, 2199, 202362, 201580, 200765, 1627742, 1629021, 1629067, 1629140, 2544, 203488, 203493, 1628393, 1626164, 1629059, 1626162, 1628994, 2037, 203933, 1629001, 204020, 1628367, 1626158, 1629028, 1628969, 1629034, 203584, 1627733, 1626196, 1628999, 2548, 1627884, 201609, 203482, 203079, 1629150, 1628389, 201949, 203585, 1626159, 202355, 201583, 2617, 1629130, 202683, 203081, 202323, 203468, 203090, 203918, 202329, 1629018, 1627774, 203086, 1627746, 1629014, 203994, 203552, 1628380, 1628369, 1627759, 202954, 202681, 1626179, 202694, 202330, 1628464, 1626154, 1627824, 203935, 1628400, 201143, 1629057, 203382, 1628408, 1626161, 1627812, 1628368, 203992, 1628412, 202692, 1627786, 1628385, 1627741, 1629117, 201147, 1628963, 203084, 201585, 1628403, 202357, 202697, 101108, 1627863, 1629109, 201569, 1629053, 201935, 101123, 203991, 200782, 203085, 202702, 2403, 1628392]

    scrapeTodaysPlayerStats(issaList)
