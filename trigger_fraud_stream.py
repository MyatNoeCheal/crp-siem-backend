"""
Calls the /fraud/simulate endpoint to replay a batch of transactions
through the live trained Autoencoder. Same pattern as seed_test_data.py.

RUN THIS ON YOUR OWN MACHINE, with your FastAPI server already running.

Usage:
    python trigger_fraud_stream.py
    python trigger_fraud_stream.py 50      # to insert 50 events instead of 20
"""

import sys
import requests

API_BASE = "http://localhost:8000"
API_KEY = "theboywholive"  # match SIEM_API_KEY on the server, or set to None

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"Requesting {count} simulated fraud events from {API_BASE}/fraud/simulate ...")

    res = requests.post(
        f"{API_BASE}/fraud/simulate",
        params={"count": count},
        headers=HEADERS,
        timeout=60,
    )
    if res.status_code == 401:
        print("FAILED (401 Unauthorized) — check API_KEY matches SIEM_API_KEY on the server.")
        return
    res.raise_for_status()
    result = res.json()
    print(f"Done: {result.get('message')}")
    print("Refresh the dashboard's Fraud tab to see the live Autoencoder-scored transactions.")


if __name__ == "__main__":
    main()