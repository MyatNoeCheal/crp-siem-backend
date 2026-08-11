import os
from pymongo import MongoClient

# Locally: uses your local MongoDB at localhost:27017 by default.
# On Render: set the MONGODB_URI environment variable to your MongoDB
# Atlas connection string (see deployment notes), and this will use
# that instead automatically -- no code change needed between environments.
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

client = MongoClient(MONGODB_URI)

db = client.crp_siem

def get_db():
    return db