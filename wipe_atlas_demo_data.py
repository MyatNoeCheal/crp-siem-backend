"""
Wipes the `logs` and `alerts` collections on your LIVE MongoDB Atlas
cluster -- reads the connection string from the MONGODB_URI environment
variable (same variable database.py and Render both use), so it will
NEVER accidentally point at localhost.

This is the ATLAS counterpart to wipe_local_demo_data.py. Keep the two
files clearly separate -- this one only runs if MONGODB_URI is actually
set in your shell, which is your safeguard against running it by
accident against nothing.

Usage (PowerShell):
    $env:MONGODB_URI="mongodb+srv://siem_admin:YOUR_ENCODED_PASSWORD@cluster0.nti8hnx.mongodb.net/?retryWrites=true&w=majority"
    python wipe_atlas_demo_data.py

Reminder: if your Atlas password contains any of these characters, they
must be percent-encoded in the connection string, or authentication will
silently fail with "bad auth":
    @ -> %40   : -> %3A   / -> %2F   # -> %23   % -> %25   ! -> %21
"""

import os
import sys
from pymongo import MongoClient

MONGODB_URI = os.environ.get("MONGODB_URI")

if not MONGODB_URI:
    print(
        "MONGODB_URI is not set in this shell -- refusing to run.\n"
        "Set it first, e.g.:\n"
        '  $env:MONGODB_URI="mongodb+srv://siem_admin:YOUR_ENCODED_PASSWORD@'
        'cluster0.nti8hnx.mongodb.net/?retryWrites=true&w=majority"'
    )
    sys.exit(1)

# Print a redacted version so you can visually confirm you're pointed at
# the right cluster without echoing your password to the terminal/logs.
_redacted = MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else MONGODB_URI
print(f"Connecting to: {_redacted}")

client = MongoClient(MONGODB_URI)
db = client.crp_siem

logs_count = db.logs.count_documents({})
alerts_count = db.alerts.count_documents({})

print("Found on Atlas:")
print(f"  logs:   {logs_count} documents")
print(f"  alerts: {alerts_count} documents")

if logs_count == 0 and alerts_count == 0:
    print("\nNothing to delete -- both collections are already empty.")
else:
    confirm = input(
        "\nThis will PERMANENTLY DELETE all documents in `logs` and `alerts`\n"
        "on your live Atlas cluster (crp_siem database). This cannot be undone.\n"
        "Type 'yes' to proceed: "
    )
    if confirm.strip().lower() == "yes":
        logs_result = db.logs.delete_many({})
        alerts_result = db.alerts.delete_many({})
        print(f"Deleted {logs_result.deleted_count} documents from logs")
        print(f"Deleted {alerts_result.deleted_count} documents from alerts")
        print(
            "\nDone. Now restart your Render service (Manual Deploy -> Restart "
            "Service, or just wait for it to spin down and cold-start again) so "
            "seed_demo_data() reseeds with mitre_attack.py + threat_intel.py "
            "enrichment."
        )
    else:
        print("Cancelled -- nothing was deleted.")