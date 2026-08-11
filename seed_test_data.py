"""
Seeds your local SIEM backend with a realistic mix of test events so all
7 dashboard tabs have something to show.

RUN THIS ON YOUR OWN MACHINE, with your FastAPI server already running
(uvicorn main:app --reload --port 8000).

Setup:
    pip install requests

If you've set SIEM_API_KEY on the server (see main.py), set the same
value below in API_KEY so this script's requests are authorized.
If you haven't set SIEM_API_KEY on the server, leave API_KEY as None
and requests will go through unauthenticated, same as before.

Usage:
    python seed_test_data.py
"""

import requests
import time
import os

API_BASE = "http://localhost:8000"

# Reads from the SIEM_API_KEY environment variable by default, so if you
# set it once in your terminal, both uvicorn and this script pick it up
# automatically. You can also just hardcode a string here instead if
# you'd rather not deal with environment variables.
API_KEY = "theboywholive"  # or: API_KEY = "your-secret-key-here"

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

events = [
    # --- Threats (failed logins, high severity, internal network) ---
    {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
     "timestamp": "2026-08-09T09:15:00", "role": "user"},
    {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
     "timestamp": "2026-08-09T09:15:10", "role": "user"},
    {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
     "timestamp": "2026-08-09T09:15:20", "role": "user"},
    {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
     "timestamp": "2026-08-09T09:15:30", "role": "user"},
    {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
     "timestamp": "2026-08-09T09:15:40", "role": "user"},  # 5th triggers the threshold rule
    {"ip": "198.51.100.23", "event_type": "port_scan", "severity": "high",
     "timestamp": "2026-08-09T10:02:00", "role": "user"},

    # --- Fraud (payments/transactions, some high-value) ---
    {"ip": "192.0.2.55", "event_type": "payment", "severity": "medium",
     "timestamp": "2026-08-09T11:00:00", "user_id": "cust_2210",
     "amount": 1899.99, "role": "user"},
    {"ip": "192.0.2.71", "event_type": "transaction", "severity": "low",
     "timestamp": "2026-08-09T11:05:00", "user_id": "cust_5521",
     "amount": 45.50, "role": "user"},
    {"ip": "192.0.2.90", "event_type": "refund", "severity": "medium",
     "timestamp": "2026-08-09T11:20:00", "user_id": "cust_3390",
     "amount": 2200.00, "role": "user"},
    {"ip": "192.0.2.14", "event_type": "chargeback", "severity": "high",
     "timestamp": "2026-08-09T11:45:00", "user_id": "cust_8871",
     "amount": 5400.00, "role": "user"},

    # --- User Behavior (logins, profile changes) ---
    {"ip": "10.0.0.5", "event_type": "login", "severity": "low",
     "timestamp": "2026-08-09T08:30:00", "user_id": "cust_1120", "role": "user"},
    {"ip": "10.0.0.6", "event_type": "profile_update", "severity": "low",
     "timestamp": "2026-08-09T08:40:00", "user_id": "cust_1121", "role": "user"},
    {"ip": "10.0.0.7", "event_type": "password_change", "severity": "medium",
     "timestamp": "2026-08-09T08:55:00", "user_id": "cust_1122", "role": "user"},
    {"ip": "10.0.0.8", "event_type": "logout", "severity": "low",
     "timestamp": "2026-08-09T09:05:00", "user_id": "cust_1120", "role": "user"},

    # --- Admin Activity ---
    {"ip": "172.16.0.2", "event_type": "admin_login", "severity": "medium",
     "timestamp": "2026-08-09T07:00:00", "user_id": "admin_greg", "role": "admin"},
    {"ip": "172.16.0.2", "event_type": "config_change", "severity": "high",
     "timestamp": "2026-08-09T07:10:00", "user_id": "admin_greg", "role": "admin"},
    {"ip": "172.16.0.3", "event_type": "role_change", "severity": "high",
     "timestamp": "2026-08-09T07:20:00", "user_id": "admin_sara", "role": "admin"},
    {"ip": "172.16.0.4", "event_type": "user_delete", "severity": "critical",
     "timestamp": "2026-08-09T07:30:00", "user_id": "admin_sara", "role": "admin"},
]


def main():
    print(f"Seeding {len(events)} events to {API_BASE}/detect ...")
    if API_KEY:
        print("Using X-API-Key header for authentication.\n")
    else:
        print("No SIEM_API_KEY set — sending requests unauthenticated.\n")

    success = 0
    for i, event in enumerate(events, 1):
        try:
            res = requests.post(f"{API_BASE}/detect", json=event, headers=HEADERS, timeout=5)
            if res.status_code == 401:
                print(f"[{i}/{len(events)}] FAILED (401 Unauthorized) — "
                      f"check that API_KEY here matches SIEM_API_KEY on the server.")
                continue
            res.raise_for_status()
            result = res.json()
            print(f"[{i}/{len(events)}] {event['event_type']:<16} "
                  f"category={result.get('category'):<16} "
                  f"risk={result.get('risk_score')} "
                  f"anomaly={result.get('anomaly')}")
            success += 1
        except Exception as e:
            print(f"[{i}/{len(events)}] FAILED ({event['event_type']}): {e}")
        time.sleep(0.1)  # small delay, avoids hammering the failed_login counter oddly

    print(f"\nDone. {success}/{len(events)} events inserted successfully.")
    print("Refresh your dashboard's Overview tab to see updated counts.")


if __name__ == "__main__":
    main()