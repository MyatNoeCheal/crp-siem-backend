"""
simulate_iot_devices.py

Simulates warehouse/fulfillment-centre IoT devices for a live SIEM demo --
NO physical hardware required. Buying and flashing real Raspberry Pi/ESP32
hardware this close to the 21 September deadline is a real risk (shipping
delays, WiFi/firmware debugging can eat days you don't have), while this
script produces the exact same live-detection demo effect: it fires HTTP
events at your SIEM's /detect endpoint, indistinguishable at the backend
from a real device doing the same thing.

WHY THIS FITS YOUR PROJECT'S SCOPE (Smart SIEM for E-Commerce): rather than
bolting on generic "IoT" as an unrelated add-on, these three simulated
devices represent warehouse/fulfillment infrastructure an e-commerce
operation would actually run -- a real, defensible attack surface for an
e-commerce SIEM to monitor, not a scope departure from your Research
Proposal.

SIMULATED DEVICES (LAN-style internal IPs, so detector.py's Rule 3
"internal network activity" bonus applies naturally, same as any real
on-prem IoT device would trigger):
    warehouse-temp-sensor-01   192.168.10.21
    smart-shelf-inventory-03   192.168.10.34
    loading-dock-cam-02        192.168.10.47

TWO MODES:

  Normal mode (default):
      python simulate_iot_devices.py
    Loops forever, sending periodic heartbeat / sensor-reading events from
    all three devices. Explicitly tagged category="user_behavior" (low
    severity) so this reads as ordinary background telemetry on the
    dashboard, not threat noise -- the same category-routing lesson your
    project already learned from the browse/add_to_cart bug in
    detector.py.

  Attack mode:
      python simulate_iot_devices.py --attack
    Runs a short burst simulating loading-dock-cam-02 being compromised:
      1. 6x failed_login events against the device's admin panel in quick
         succession -- this deliberately reuses your EXISTING, already-
         verified failed-login threshold rule in detector.py
         (FAILED_LOGIN_THRESHOLD=5 within FAILED_LOGIN_WINDOW_MINUTES=10),
         so the alert you see fire is your real detection logic working,
         not a special-cased demo path.
      2. A firmware-tampering event (critical severity).
      3. A command-injection attempt (high severity).
      4. A DDoS-participation event (critical severity) -- the compromised
         camera now flooding outbound traffic, e.g. joined to a botnet.
    Then exits (does not loop) -- run this live during your demo, right
    after showing a few minutes of normal-mode traffic on the dashboard.

USAGE FOR A LIVE DEMO:
    1. Start normal mode in one terminal, let it run for a minute or two
       so the dashboard shows quiet background IoT telemetry.
    2. Ctrl+C it (or leave it running in a second terminal).
    3. Run `python simulate_iot_devices.py --attack` and watch the
       Threats tab -- the failed-login alert should appear within
       seconds, followed by the critical firmware/command-injection/DDoS
       events.

Requires: pip install requests
"""

import argparse
import random
import time
from datetime import datetime, timezone

import requests

# =========================
# CONFIGURATION
# =========================
# Points at the live deployment by default -- this is what your dashboard
# actually watches, so a demo run here is what your audience will see.
# Override with --local to hit a locally-run backend instead.
SIEM_API_BASE_LIVE = "https://crp-siem-backend.onrender.com"
SIEM_API_BASE_LOCAL = "http://localhost:8000"
SIEM_API_KEY_LIVE = "7777"
SIEM_API_KEY_LOCAL = "theboywholive"

DEVICES = [
    {"id": "warehouse-temp-sensor-01", "ip": "192.168.10.21"},
    {"id": "smart-shelf-inventory-03", "ip": "192.168.10.34"},
    {"id": "loading-dock-cam-02", "ip": "192.168.10.47"},
]

NORMAL_INTERVAL_SECONDS = 8  # how often each device "checks in" in normal mode


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def send_event(base_url, api_key, event):
    event.setdefault("timestamp", now_iso())
    try:
        resp = requests.post(
            f"{base_url}/detect",
            json=event,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=5,
        )
        status = resp.status_code
        marker = "OK" if status < 400 else f"HTTP {status}"
        print(f"  -> {event['event_type']:<24} [{event.get('severity','?'):<8}] {marker}")
    except requests.RequestException as e:
        print(f"  -> {event['event_type']:<24} FAILED to send: {e}")


def normal_event(device):
    """
    Ordinary telemetry -- explicit category="user_behavior" so this stays
    out of the Threats tab, matching how real background device activity
    should read on a SIEM dashboard.
    """
    kind = random.choice(["device_heartbeat", "sensor_reading"])
    event = {
        "event_type": kind,
        "severity": "low",
        "ip": device["ip"],
        "user_id": device["id"],
        "role": "device",
        "category": "user_behavior",
    }
    if kind == "sensor_reading":
        event["reason"] = [f"temperature={round(random.uniform(2.0, 8.0), 1)}C"] \
            if "temp" in device["id"] else [f"stock_weight={round(random.uniform(10, 500), 1)}kg"]
    return event


def run_normal_mode(base_url, api_key):
    print(f"Simulating {len(DEVICES)} IoT devices against {base_url} (Ctrl+C to stop)...\n")
    try:
        while True:
            device = random.choice(DEVICES)
            send_event(base_url, api_key, normal_event(device))
            time.sleep(NORMAL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_attack_mode(base_url, api_key):
    target = next(d for d in DEVICES if d["id"] == "loading-dock-cam-02")
    print(f"Simulating a compromise of {target['id']} ({target['ip']}) against {base_url}...\n")

    print("Step 1: brute-forcing the device's admin panel (reuses your existing failed_login rule)")
    for i in range(6):
        send_event(base_url, api_key, {
            "event_type": "failed_login",
            "severity": "high",
            "ip": target["ip"],
            "user_id": target["id"],
            "role": "device",
        })
        time.sleep(1.5)

    print("\nStep 2: firmware tampering detected")
    send_event(base_url, api_key, {
        "event_type": "device_firmware_tampering",
        "severity": "critical",
        "ip": target["ip"],
        "user_id": target["id"],
        "role": "device",
    })
    time.sleep(1)

    print("\nStep 3: command injection attempt against the device's control interface")
    send_event(base_url, api_key, {
        "event_type": "device_command_injection",
        "severity": "high",
        "ip": target["ip"],
        "user_id": target["id"],
        "role": "device",
    })
    time.sleep(1)

    print("\nStep 4: compromised device now flooding outbound traffic (botnet participation)")
    send_event(base_url, api_key, {
        "event_type": "device_ddos_participation",
        "severity": "critical",
        "ip": target["ip"],
        "user_id": target["id"],
        "role": "device",
    })

    print("\nDone. Check the dashboard's Threats and Logs tabs -- the failed-login burst "
          "should already have produced a High-severity alert, and the critical events "
          "should appear in Logs even without a dedicated IoT category.")


def main():
    parser = argparse.ArgumentParser(description="Simulate IoT device events for the Smart SIEM demo.")
    parser.add_argument("--attack", action="store_true", help="Run the attack-burst scenario once, then exit.")
    parser.add_argument("--local", action="store_true", help="Target a locally-running backend instead of Render.")
    args = parser.parse_args()

    base_url = SIEM_API_BASE_LOCAL if args.local else SIEM_API_BASE_LIVE
    api_key = SIEM_API_KEY_LOCAL if args.local else SIEM_API_KEY_LIVE

    if args.attack:
        run_attack_mode(base_url, api_key)
    else:
        run_normal_mode(base_url, api_key)


if __name__ == "__main__":
    main()