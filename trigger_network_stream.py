"""
Calls the /threats/simulate-network endpoint to replay a batch of
UNSW-NB15 test-set flows through the live trained Network Intrusion
Autoencoder. Same pattern as trigger_fraud_stream.py / seed_test_data.py.

RUN THIS ON YOUR OWN MACHINE, with your FastAPI server already running.

Usage:
    python trigger_network_stream.py
    python trigger_network_stream.py 50      # to insert 50 events instead of 20
"""

import sys
import requests

API_BASE = "http://localhost:8000"
API_KEY = "theboywholive"  # match SIEM_API_KEY on the server, or set to None

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"Requesting {count} simulated network intrusion events from "
          f"{API_BASE}/threats/simulate-network ...")

    res = requests.post(
        f"{API_BASE}/threats/simulate-network",
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
    print("Refresh the dashboard's Threats tab to see the live "
          "Autoencoder-scored network flows.")


if __name__ == "__main__":
    main()