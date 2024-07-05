# This file works with Azure client libraries in python to store
# the user conversations to Azure.
import os
import urllib.parse
import pymongo
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = os.environ.get("MONGO_CLIENT_STRING")
USERNAME = urllib.parse.quote_plus("tejaschaudhari1")
PASSWORD = urllib.parse.quote_plus("Insights@Wepa")

def get_client():
    return pymongo.MongoClient("mongodb+srv://%s:%s@insights-generative.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" % (USERNAME, PASSWORD))

client = get_client()

db = client.get_database("BoardEfficiencyCompanion")

collection = db.get_collection("ChatHistory")
