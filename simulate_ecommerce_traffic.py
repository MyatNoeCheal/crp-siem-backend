"""
Simulates realistic e-commerce site traffic hitting your live /detect
endpoint -- the same shape of requests siem_event_sender.php will send
once it's wired into the real PHP site. Use this to verify the whole
pipeline (rule engine -> alert creation -> MITRE/threat-intel enrichment
-> dashboard) BEFORE touching production.

Unlike seed_test_data.py (which seeds a fixed, hand-written batch), this
generates a randomized, timestamped SESSION-shaped stream: normal
customer browsing/checkout mixed with a couple of deliberately suspicious
patterns (a brute-force burst, one high-value transaction), so you can
watch the dashboard react to something that looks like real traffic
rather than a static fixture.

RUN THIS ON YOUR OWN MACHINE, against either your local server or the
live Render deployment.

Setup:
    pip install requests

Usage:
    python simulate_ecommerce_traffic.py                     # local, 15 customer sessions
    python simulate_ecommerce_traffic.py --sessions 30
    python simulate_ecommerce_traffic.py --api-base https://your-render-app.onrender.com
"""

import argparse
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "theboywholive"  # match SIEM_API_KEY on the server, or pass --api-key ""

CUSTOMER_IP_POOL = [f"81.171.{a}.{b}" for a in range(10, 30) for b in range(10, 30)]
ADMIN_IPS = ["172.16.0.2", "172.16.0.3"]

PRODUCT_PRICES = [19.99, 34.50, 89.00, 129.00, 249.99, 599.00, 1249.00, 1899.99]


def _ts(minutes_ago):
    # Timezone-aware, then drop tzinfo -- matches the naive-UTC ISO strings
    # every other timestamp in this codebase (main.py, seed_test_data.py,
    # etc.) already produces, so this stays consistent with the rest of
    # the pipeline rather than introducing a "+00:00" suffix on its own.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - timedelta(minutes=minutes_ago)).isoformat()


def _post(session, api_base, headers, event, label):
    try:
        res = session.post(f"{api_base}/detect", json=event, headers=headers, timeout=8)
        if res.status_code == 401:
            print(f"  [FAIL] {label}: 401 Unauthorized -- check --api-key matches SIEM_API_KEY")
            return
        res.raise_for_status()
        result = res.json()
        flag = "ANOMALY" if result.get("anomaly") else "ok"
        print(f"  [{flag:7}] {label:<28} risk={result.get('risk_score'):>3}  "
              f"category={result.get('category')}")
    except Exception as e:
        print(f"  [ERROR ] {label}: {e}")


def normal_customer_session(session, api_base, headers, minutes_ago_start):
    """A believable browse -> cart -> checkout -> logout session."""
    user_id = f"cust_{random.randint(1000, 9999)}"
    ip = random.choice(CUSTOMER_IP_POOL)
    t = minutes_ago_start

    steps = [
        ("login", "low", None),
        ("browse", "low", None),
        ("browse", "low", None),
        ("add_to_cart", "low", None),
        ("checkout", "low", None),
        ("payment", "medium", round(random.choice(PRODUCT_PRICES), 2)),
        ("logout", "low", None),
    ]

    for event_type, severity, amount in steps:
        event = {
            "ip": ip,
            "event_type": event_type,
            "severity": severity,
            "timestamp": _ts(t),
            "user_id": user_id,
            "role": "user",
        }
        if amount is not None:
            event["amount"] = amount
        _post(session, api_base, headers, event, f"{user_id} {event_type}")
        t -= random.uniform(0.5, 2.0)

    return t


def brute_force_burst(session, api_base, headers, minutes_ago_start):
    """5 failed logins from one IP within the 10-min detector window --
    should trip detector.py's FAILED_LOGIN_THRESHOLD rule."""
    ip = f"185.220.{random.randint(100,199)}.{random.randint(1,254)}"
    print(f"\n-- simulating brute-force burst from {ip} --")
    t = minutes_ago_start
    for i in range(5):
        event = {
            "ip": ip,
            "event_type": "failed_login",
            "severity": "high",
            "timestamp": _ts(t),
            "role": "user",
        }
        _post(session, api_base, headers, event, f"failed_login #{i+1} from {ip}")
        t -= 1.0
    return t


def high_value_transaction(session, api_base, headers, minutes_ago_start):
    """One transaction well above typical order size -- should trip the
    dynamic 90th-percentile high-value rule once enough history exists."""
    user_id = f"cust_{random.randint(1000, 9999)}"
    ip = random.choice(CUSTOMER_IP_POOL)
    event = {
        "ip": ip,
        "event_type": "payment",
        "severity": "medium",
        "timestamp": _ts(minutes_ago_start),
        "user_id": user_id,
        "amount": round(random.uniform(3000, 6000), 2),
        "role": "user",
    }
    _post(session, api_base, headers, event, f"HIGH-VALUE payment {user_id}")
    return minutes_ago_start - 1


def admin_action(session, api_base, headers, minutes_ago_start):
    admin_id = f"admin_{random.choice(['greg', 'sara', 'lin'])}"
    ip = random.choice(ADMIN_IPS)
    event_type = random.choice(["admin_login", "config_change", "role_change"])
    event = {
        "ip": ip,
        "event_type": event_type,
        "severity": "medium" if event_type == "admin_login" else "high",
        "timestamp": _ts(minutes_ago_start),
        "user_id": admin_id,
        "role": "admin",
    }
    _post(session, api_base, headers, event, f"{admin_id} {event_type}")
    return minutes_ago_start - 1


def main():
    parser = argparse.ArgumentParser(description="Simulate e-commerce traffic against /detect")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                         help="Pass an empty string if SIEM_API_KEY isn't set on the server")
    parser.add_argument("--sessions", type=int, default=15, help="Number of normal customer sessions")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    session = requests.Session()

    print(f"Target: {args.api_base}/detect")
    print(f"Simulating {args.sessions} normal customer sessions + 1 brute-force burst "
          f"+ 1 high-value transaction + a few admin actions.\n")

    t = args.sessions * 3  # spread sessions backward in time so it looks organic

    print("-- normal customer sessions --")
    for _ in range(args.sessions):
        t = normal_customer_session(session, args.api_base, headers, t)
        time.sleep(0.05)

    t = brute_force_burst(session, args.api_base, headers, t)

    print("\n-- high-value transaction --")
    t = high_value_transaction(session, args.api_base, headers, t)

    print("\n-- admin actions --")
    for _ in range(3):
        t = admin_action(session, args.api_base, headers, t)

    print("\nDone. Check the dashboard:")
    print("  Overview  -> event counts / timeline should reflect this run")
    print("  Threats   -> the brute-force burst should appear as a grouped alert")
    print("  Fraud     -> the high-value payment should show elevated risk_score")
    print("  Admin Activity -> the 3 admin actions")
    print("\nIf everything looks right here, siem_event_sender.php can point at the")
    print("same --api-base and SIEM_API_KEY with confidence.")


if __name__ == "__main__":
    main()