import csv
from pymongo import MongoClient
import glob


#connect to the database
client = MongoClient('mongodb+srv://rohanrao35:Npsnps407407@cluster0-8eolw.mongodb.net/test?retryWrites=true')
db = client["NPS"]


def storePlayerStats():
    col = db["PLAYER_STATS"]
    for filename in glob.glob('./datasets/player_stats/*.csv'):
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', strict=True)
            line_count = 0
            playerObj = {}
            for row in csv_reader:
                if line_count == 0:
                    #This is the header of the csv_file
                    #print(f'Coloumn names {", ".join(row)}')
                    line_count += 1
                    continue
                elif line_count == 1:
                    playerObj["_id"] = row[0]
                    player_name = filename.split('/')[-1]
                    player_name = player_name[:-10]
                    player_name = player_name[13:]
                    #print(player_name)
                    playerObj["playerName"] = player_name
                    playerObj["seasons"] = []
                i = 3
                season = {

                    "LEAGUE_ID": row[2],
                    "TEAM_ID": row[3],
                    "TEAM_ABBREVIATION": row[4],
                    "PLAYER_AGE": row[5],
                    "GP": row[6],
                    "GS": row[7],
                    "MIN": row[8],
                    "FGM": row[9],
                    "FGA": row[10],
                    "FG_PCT": row[11],
                    "FG3M": row[12],
                    "FG3A": row[13],
                    "FG3_PCT": row[14],
                    "FTM": row[15],
                    "FTA": row[16],
                    "FT_PCT": row[17],
                    "OREB": row[18],
                    "DREB": row[19],
                    "REB": row[20],
                    "AST": row[21],
                    "STL": row[22],
                    "BLK": row[23],
                    "TOV": row[24],
                    "PF": row[25],
                    "PTS": row[26]
                }
                playerObj["seasons"].append(season)
                line_count += 1

            if playerObj:
                # print(row[i])
                # i += 1
                # #print("\n")
                # print(row[i])
                # i += 1
                # #print("\n")
                # print(row[i])
                # i += 1
                # #print("\n")
                # print(row[i])
                # i += 1
                #print("\n")
                #exit(1)
                print(playerObj)

                #exit(1)
                col.replace_one({'_id':playerObj['_id']}, playerObj, upsert=True)


def storeTeamStats():
    col = db["TEAM_STATS"]
    for filename in glob.glob('./datasets/team_stats/*.csv'):
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', strict=True)
            teamObj = {}
            line_count = 0
            for row in csv_file:
                #print(row)
                if line_count == 0:
                    #print(f'Coloumn names {", ".join(row)}')
                    print(row)
                    line_count += 1
                    teamObj["years"] = []
                    continue;
                elif line_count != 1:
                    row = row.split(',')
                    print(row)
                    teamObj["_id"] = row[0]
                    teamObj["teamCity"] = row[1]
                    teamObj["teamName"] = row[2]
                    #teamObj["years"] = []
                    year = {
                        "YEAR": row[3],
                        "GP": row[4],
                        "WINS": row[5],
                        "LOSSES": row[6],
                        "WIN_PCT": row[7],
                        "CONF_RANK": row[8],
                        "DIV_RANK": row[9],
                        "PO_WINS": row[10],
                        "PO_LOSSES": row[11],
                        "CONF_COUNT": row[12],
                        "DIV_COUNT": row[13],
                        "NBA_FINALS_APPEARANCE": row[14],
                        "FGM": row[15],
                        "FGA": row[16],
                        "FG_PCT": row[17],
                        "FG3M": row[18],
                        "FG3A": row[19],
                        "FG3_PCT": row[20],
                        "FTM": row[21],
                        "FTA": row[22],
                        "FT_PCT": row[23],
                        "OREB": row[24],
                        "DREB": row[25],
                        "REB": row[26],
                        "AST": row[27],
                        "PF": row[28],
                        "STL": row[29],
                        "TOV": row[30],
                        "BLK": row[31],
                        "PTS": row[32],
                        "PTS_RANK": row[33]
                    }
                    # print(teamObj["_id"])
                    # print(teamObj["teamCity"])
                    # print(teamObj["teamName"])
                    teamObj["years"].append(year)
                else:
                    row = row.split(',')
                    #print(row[1])
                    #print(row[3])
                    year = {
                        "YEAR": row[3],
                        "GP": row[4],
                        "WINS": row[5],
                        "LOSSES": row[6],
                        "WIN_PCT": row[7],
                        "CONF_RANK": row[8],
                        "DIV_RANK": row[9],
                        "PO_WINS": row[10],
                        "PO_LOSSES": row[11],
                        "CONF_COUNT": row[12],
                        "DIV_COUNT": row[13],
                        "NBA_FINALS_APPEARANCE": row[14],
                        "FGM": row[15],
                        "FGA": row[16],
                        "FG_PCT": row[17],
                        "FG3M": row[18],
                        "FG3A": row[19],
                        "FG3_PCT": row[20],
                        "FTM": row[21],
                        "FTA": row[22],
                        "FT_PCT": row[23],
                        "OREB": row[24],
                        "DREB": row[25],
                        "REB": row[26],
                        "AST": row[27],
                        "PF": row[28],
                        "STL": row[29],
                        "TOV": row[30],
                        "BLK": row[31],
                        "PTS": row[32],
                        "PTS_RANK": row[33]
                    }
                    teamObj["teamCity"] = row[1]
                    teamObj["teamName"] = row[2]
                    teamObj["years"].append(year)
                line_count += 1
            if teamObj:
                #print(teamObj)
                col.replace_one({'_id':teamObj['_id']}, teamObj, upsert=True)

def storeBoxScores():
    col = db["BOX_SCORES"]
    col2 = db["TEAM_ROSTERS"]
    for filename in glob.glob('./datasets/*.csv'):
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', strict=True)
            teamObj = {}
            rosterObj = {}
            line_count = 0
            for row in csv_file:
                if line_count == 0:
                    line_count += 1
                    teamObj["games"] = []
                    #print(row)
                else:
                    row = row.split(',')
                    #print(row[1])
                    teamObj["_id"] = row[1]
                    teamObj["TEAM_ABBREVIATION"] = row[2]
                    teamObj["TEAM_NAME"] = row[3]
                    rosterObj["TEAM_NAME"] = row[3]
                    rosterObj["_id"] = row[1]
                    game = {
                        "SEASON_ID": row[0],
                        "GAME_ID": row[4],
                        "GAME_DATE": row[5],
                        "MATCHUP": row[6],
                        "WL": row[7],
                        "MIN": row[8],
                        "PTS": row[9],
                        "FGM": row[10],
                        'FGA': row[11],
                        "FG_PCT": row[12],
                        "FG3M": row[13],
                        "FG3A": row[14],
                        "FG3_PCT": row[15],
                        "FTM": row[16],
                        "FTA": row[17],
                        "FT_PCT": row[18],
                        "OREB": row[19],
                        "DREB": row[20],
                        "REB": row[21],
                        "AST": row[22],
                        "STL": row[23],
                        "BLK": row[24],
                        "TOV": row[25],
                        "PF": row[26],
                        "PLUS_MINUS": row[27],
                        "NUM_WINS": row[28],
                        "NUM_LOSSES": row[29][:-1]
                    }
                    teamObj["games"].append(game)
                    #print(teamObj)

            if teamObj:
                col.replace_one({'_id':teamObj['_id']}, teamObj, upsert=True)
                col2.replace_one({'_id':teamObj['_id']}, rosterObj, upsert=True)
            # print(teamObj["TEAM_NAME"])
            # print(teamObj["TEAM_ABBREVIATION"])
            # print(teamObj["_id"])
            # print(teamObj["games"][0])




if __name__ == "__main__":
    #storePlayerStats()
    #storeTeamStats()
    storeBoxScores()
    storeTeamRosters()
