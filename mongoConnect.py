import pymongo


client = pymongo.MongoClient("mongodb+srv://rohanrao35:Npsnps407407@cluster0-8eolw.mongodb.net/test?retryWrites=true")
db = client["NPS"]
col = db["Test"]

#Code to insert
# mydict = { "name": "John", "address": "Highway 37" }
#
# x = col.insert_one(mydict)





#Code to query
# mydoc = col.find({'ID': 1})
#
# for x in mydoc:
#   print(x)
