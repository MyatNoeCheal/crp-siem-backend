"""
Wipes the `logs` and `alerts` collections on your LOCAL MongoDB
(mongodb://localhost:27017, same default database.py falls back to when
MONGODB_URI isn't set) so the next uvicorn startup's seed_demo_data()
reseeds from scratch -- with mitre_attack.py + threat_intel.py
enrichment included this time.

This is the local equivalent of wipe_atlas_demo_data.py -- same idea,
just pointed at your local Mongo instead of Atlas. Useful when you don't
have the mongosh shell installed.

Usage:
    python wipe_local_demo_data.py
"""

from pymongo import MongoClient

MONGODB_URI = "mongodb://localhost:27017"

client = MongoClient(MONGODB_URI)
db = client.crp_siem

logs_count = db.logs.count_documents({})
alerts_count = db.alerts.count_documents({})

print(f"Connecting to: {MONGODB_URI}")
print("Found locally:")
print(f"  logs:   {logs_count} documents")
print(f"  alerts: {alerts_count} documents")

if logs_count == 0 and alerts_count == 0:
    print("\nNothing to delete -- both collections are already empty.")
else:
    confirm = input(
        "\nThis will PERMANENTLY DELETE all documents in `logs` and `alerts`\n"
        "on your LOCAL MongoDB (crp_siem database). This cannot be undone.\n"
        "Type 'yes' to proceed: "
    )
    if confirm.strip().lower() == "yes":
        logs_result = db.logs.delete_many({})
        alerts_result = db.alerts.delete_many({})
        print(f"Deleted {logs_result.deleted_count} documents from logs")
        print(f"Deleted {alerts_result.deleted_count} documents from alerts")
        print(
            "\nDone. Now restart uvicorn (Ctrl+C, then "
            "`uvicorn main:app --reload --port 8000`) so seed_demo_data() "
            "reseeds with mitre_attack.py + threat_intel.py enrichment."
        )
    else:
        print("Cancelled -- nothing was deleted.")