"""
Enriches an IP address with geolocation + reputation context, the way a
real SIEM (Splunk ES, QRadar, Sentinel) enriches every event with "where
is this IP, and is it already known-bad" before an analyst ever looks at
it -- turning a bare IP string into something actionable.

TWO PIECES:
  1. GEOLOCATION -- via ip-api.com's free endpoint (no API key required,
     45 requests/minute limit, HTTP only on the free tier). Results are
     cached in MongoDB (ip_intel_cache) for 24h so repeat lookups of the
     same IP (very common -- the same attacker IP shows up dozens of
     times) don't re-hit the rate limit or add latency to /detect.
  2. REPUTATION -- checks against a small static DEMONSTRATION blocklist
     by default. If an ABUSEIPDB_API_KEY environment variable is set,
     uses the real AbuseIPDB free-tier API (1,000 checks/day) instead.

HONEST SCOPE NOTE (state this in the report, same pattern as the other
honesty notes in this codebase): production SIEMs pay for commercial
threat-intel feeds (Recorded Future, VirusTotal Enterprise, AbuseIPDB
Pro) with millions of continuously-updated indicators. This project uses
free/no-key geolocation and either a small static demonstration
blocklist or AbuseIPDB's free tier -- enough to demonstrate the
enrichment *pattern* a real SIEM uses, not production-grade coverage.
Say this plainly rather than implying enterprise-grade threat intel.

NOTE ON DEMO/SIMULATED IPs: the demo seed data and simulated streams in
this project use RFC 5737 documentation-reserved ranges (192.0.2.0/24,
198.51.100.0/24, 203.0.113.0/24) -- these deliberately do NOT resolve to
a real geographic location (ip-api.com will return a "reserved range"
failure for them), which is expected and worth noting in the report:
geolocation is only meaningful for real public traffic. The demo
blocklist below is matched against the SAME IPs already used in
main.py's _demo_events() (203.0.113.14, 198.51.100.23), so the
REPUTATION half of enrichment still demos correctly even though the
GEO half won't for reserved-range demo IPs.

Private/reserved IP ranges (RFC 1918, loopback, link-local) are
recognized locally and never sent externally -- both because
geolocation is meaningless for them and to avoid leaking internal
demo/test traffic to a third-party API.
"""

import os
import ipaddress
import time
from datetime import datetime, timedelta

import requests

from database import get_db

db = get_db()
_cache_collection = db.ip_intel_cache

CACHE_TTL_HOURS = 24
_ABUSEIPDB_KEY = os.environ.get("ABUSEIPDB_API_KEY")

# Small, clearly-labelled DEMONSTRATION blocklist -- illustrative only,
# NOT a real threat feed. Deliberately matches the IPs already used in
# main.py's _demo_events() (failed_login / port_scan sources) so the
# seeded demo data shows a "malicious" reputation badge out of the box.
_DEMO_BLOCKLIST = {
    "203.0.113.14": "Known brute-force source (demo blocklist)",
    "198.51.100.23": "Known scanner (demo blocklist)",
    "185.220.101.1": "Tor exit node (demo blocklist)",
    "45.155.205.1": "Known malware C2 (demo blocklist)",
}


def _is_private(ip):
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        return True  # unparseable -- treat as non-lookupable rather than erroring


def _from_cache(ip):
    doc = _cache_collection.find_one({"ip": ip})
    if not doc:
        return None
    cached_at = doc.get("cached_at")
    if not cached_at:
        return None
    try:
        cached_dt = datetime.fromisoformat(cached_at)
    except Exception:
        return None
    if datetime.utcnow() - cached_dt > timedelta(hours=CACHE_TTL_HOURS):
        return None  # stale -- treat as a cache miss, will be refreshed
    doc.pop("_id", None)
    doc.pop("cached_at", None)
    doc.pop("ip", None)
    return doc


def _save_cache(ip, data):
    _cache_collection.update_one(
        {"ip": ip},
        {"$set": {**data, "ip": ip, "cached_at": datetime.utcnow().isoformat()}},
        upsert=True,
    )


def _geolocate(ip):
    """Free ip-api.com lookup. Returns None on failure -- enrichment is a
    nice-to-have, not something that should break /detect if the
    third-party API is slow, down, rate-limited, or the IP is a
    reserved/documentation range (see module docstring)."""
    try:
        res = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,countryCode,city,lat,lon,isp,org,as"},
            timeout=2,
        )
        data = res.json()
        if data.get("status") != "success":
            return None
        return {
            "country": data.get("country"),
            "country_code": data.get("countryCode"),
            "city": data.get("city"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "isp": data.get("isp"),
            "org": data.get("org"),
            "asn": data.get("as"),
        }
    except Exception:
        return None


def _check_abuseipdb(ip):
    try:
        res = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers={"Key": _ABUSEIPDB_KEY, "Accept": "application/json"},
            timeout=2,
        )
        data = res.json().get("data", {})
        score = data.get("abuseConfidenceScore", 0)
        if score >= 50:
            return {"reputation": "malicious", "reputation_reason": f"AbuseIPDB confidence {score}%", "abuse_score": score}
        elif score >= 10:
            return {"reputation": "suspicious", "reputation_reason": f"AbuseIPDB confidence {score}%", "abuse_score": score}
        return {"reputation": "clean", "reputation_reason": None, "abuse_score": score}
    except Exception:
        return None


def _check_reputation(ip):
    if _ABUSEIPDB_KEY:
        result = _check_abuseipdb(ip)
        if result:
            return result
    # Fallback: static demo blocklist (see module docstring)
    if ip in _DEMO_BLOCKLIST:
        return {"reputation": "malicious", "reputation_reason": _DEMO_BLOCKLIST[ip], "abuse_score": None}
    return {"reputation": "unknown", "reputation_reason": None, "abuse_score": None}


def enrich_ip(ip):
    """
    Returns enrichment context for an IP:
        {
          "ip": str, "is_private": bool,
          "country": str | None, "country_code": str | None,
          "city": str | None, "lat": float | None, "lon": float | None,
          "isp": str | None, "asn": str | None,
          "reputation": "malicious" | "suspicious" | "clean" | "unknown" | "internal",
          "reputation_reason": str | None,
        }
    Never raises -- worst case returns a minimal dict with nulls, so an
    enrichment failure never breaks the calling /detect request.
    """
    if not ip:
        return {"ip": ip, "is_private": True, "reputation": "unknown"}

    if _is_private(ip):
        return {
            "ip": ip, "is_private": True,
            "country": None, "country_code": None, "city": None,
            "lat": None, "lon": None, "isp": None, "asn": None,
            "reputation": "internal", "reputation_reason": "Private/internal network range",
        }

    cached = _from_cache(ip)
    if cached:
        return {"ip": ip, "is_private": False, **cached}

    geo = _geolocate(ip) or {}
    rep = _check_reputation(ip)

    result = {**geo, **rep}
    _save_cache(ip, result)

    return {"ip": ip, "is_private": False, **result}


def enrich_ips_batch(ips, throttle_seconds=0.05):
    """
    Convenience helper for enriching several IPs at once (e.g. the
    Overview 'Top Offenders' panel). Only sleeps between calls that
    actually hit the external API (private IPs and cache hits are free),
    to stay under ip-api.com's 45 req/min free-tier limit without
    slowing down a mostly-cached request.
    """
    results = {}
    for ip in ips:
        already_cached = (not _is_private(ip)) and (_from_cache(ip) is not None)
        results[ip] = enrich_ip(ip)
        if not results[ip].get("is_private") and not already_cached:
            time.sleep(throttle_seconds)
    return results