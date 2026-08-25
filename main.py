# uvicorn main:app --reload --port 8000

from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from bson import ObjectId
from typing import Optional, List
from datetime import datetime, timedelta
import os
import json
import math

from ai.detector import AnomalyDetector
from models import LogData, LogEvent
from database import get_db
from utils import normalize_log, get_risk_level
from dataset.scripts.clean_email import get_logs
from ai_insights import generate_insights
from lstm_inference import score_user_sequence
from fraud_stream import generate_simulated_fraud_events
from network_stream import generate_simulated_network_events
from mitre_attack import get_mitre_mapping
from threat_intel import enrich_ip
from entity_risk import update_entity_risk, get_entity_risk, get_top_risk_entities
import cases as cases_module
import webhooks as webhooks_module
import auth

import dataset.scripts.clean_email as ce
from entity_risk import update_entity_risk, get_entity_risk, get_top_risk_entities, get_escalating_entities

print("LOADED FILE:", ce.__file__)
print("AVAILABLE FUNCTIONS:", dir(ce))


app = FastAPI()
detector = AnomalyDetector()

# =========================
# CORS
# =========================
# Add the e-commerce site's domain (and localhost for dev) here.
# Wildcard "*" is fine for local prototyping but should be replaced
# with explicit origins before the demo/deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://your-ecommerce-site.com", "http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_db()
collection = db.email_logs
logs_collection = db.logs
alerts_collection = db.alerts

detector.db = db  # gives the detector access to Mongo for the dynamic
                  # high-value-transaction threshold (see detector.py)


# =========================
# ALERT LIFECYCLE — persistent, triageable alerts (New / Investigating /
# Resolved / False Positive), the way a real SIEM separates "raw log
# events" (every event, in logs_collection, visible on the Logs tab) from
# "alerts" (only the anomalous ones that actually need an analyst's
# attention, in alerts_collection, visible on the Threats tab).
#
# This REPLACES the old on-the-fly _correlate_incidents() grouping (still
# defined below, now unused, kept for reference): that function recomputed
# groupings from raw logs on every /threats request and had no memory of
# anything -- an analyst's triage decision would just vanish on the next
# refresh. Alerts persist in their own collection instead, so status
# survives across requests exactly like a real alert queue.
# =========================
ALERT_STATUSES = {"new", "investigating", "resolved", "false_positive"}

# How close together repeated events of the same type/source have to be
# to merge into one alert instead of creating a new one -- matches the
# window _correlate_incidents() used to use, so alert grouping behavior
# is unchanged from before this refactor.
ALERT_CORRELATION_WINDOW_MINUTES = 10


class AlertStatusUpdate(BaseModel):
    status: str
    note: Optional[str] = None


# =========================
# ANALYST AUTHENTICATION + RBAC — see auth.py for the full design.
# Login is stateless JWT; two roles ("analyst", "admin"). This REPLACES
# the "no login flow exists" reasoning that previously left alert
# triage / case management / webhook endpoints unprotected below --
# every one of those now requires a valid analyst session, and a few
# (webhook deletion, user management) additionally require "admin".
# =========================
class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "analyst"
    display_name: Optional[str] = None


class UserActiveUpdate(BaseModel):
    active: bool


# =========================
# CASE MANAGEMENT — turns the alert queue into an actual investigation
# workflow (see cases.py for the full design rationale). A case groups
# one or more related alerts, carries an assignment + status of its own,
# and keeps an append-only timeline of notes/status changes/links, the
# way a real SOC separates "detection engine output" (alerts) from
# "analyst investigation record" (cases).
#
# Same as /alerts/{id}/status above, these endpoints are deliberately
# NOT behind require_api_key -- they're triggered by an analyst working
# inside the dashboard itself, which has no login/key-entry flow. In a
# production deployment these would sit behind real analyst
# authentication instead.
# =========================
class CaseCreate(BaseModel):
    title: str
    alert_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    tags: Optional[List[str]] = None


class CaseAlertLink(BaseModel):
    alert_id: str


class CaseNoteCreate(BaseModel):
    content: str


# =========================
# WEBHOOK NOTIFICATIONS — external systems (Slack, a ticketing tool, a
# custom listener) subscribe to SIEM events instead of polling the API.
# See webhooks.py for the full design, event types, and the honest
# single-attempt-delivery limitation. Same no-API-key rationale as the
# alert/case endpoints above -- this is an analyst-configured integration
# via the dashboard, not external ingestion traffic.
# =========================
class WebhookCreate(BaseModel):
    url: str
    events: List[str]
    secret: Optional[str] = None
    description: Optional[str] = None
    active: bool = True


class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    description: Optional[str] = None
    active: Optional[bool] = None


def _upsert_alert(event_dict, score, level, is_anomaly, reasons, category,
                   mitre=None, threat_intel=None):
    """
    Creates or updates a persistent alert for an anomalous event.

    Only called when is_anomaly=True -- a real SIEM doesn't raise an
    alert for every log line, only for things that cross a risk
    threshold. Non-anomalous events are still stored in logs_collection
    (visible on the Logs tab) but never become alerts.

    Correlation/deduplication: if an OPEN alert (status new/investigating)
    already exists for the same ip + event_type + category within the
    last ALERT_CORRELATION_WINDOW_MINUTES, this event is merged into it
    (count incremented, risk re-maxed, reasons merged) instead of creating
    a duplicate alert. A resolved/false-positive alert is deliberately
    NOT matched here -- once an analyst has closed it, a new occurrence
    should open a fresh alert rather than silently reopening a closed one.

    mitre / threat_intel: optional enrichment dicts (see mitre_attack.py
    and threat_intel.py) attached to the alert for display on the
    Threats/Admin Activity tabs. On merge into an existing alert, a
    freshly-provided value overwrites the stored one; None preserves
    whatever the alert already had (so a caller that doesn't compute
    enrichment, e.g. the simulated streams, doesn't blank out earlier
    enrichment on merge).

    Returns the alert's _id (new or existing), or None if not anomalous.
    """
    if not is_anomaly:
        return None

    timestamp = event_dict.get("timestamp")
    window_start_iso = None
    if timestamp:
        try:
            ts_dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            window_start_iso = (ts_dt - timedelta(minutes=ALERT_CORRELATION_WINDOW_MINUTES)).isoformat()
        except Exception:
            window_start_iso = None

    query = {
        "ip": event_dict.get("ip"),
        "event_type": event_dict.get("event_type"),
        "category": category,
        "status": {"$in": ["new", "investigating"]},
    }
    if window_start_iso:
        query["last_seen"] = {"$gte": window_start_iso}

    existing = alerts_collection.find_one(query, sort=[("last_seen", -1)])

    if existing:
        merged_reasons = list(existing.get("reason", []))
        for r in reasons:
            if r not in merged_reasons:
                merged_reasons.append(r)
        new_score = max(existing.get("risk_score", 0), score)

        alerts_collection.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "last_seen": timestamp,
                    "risk_score": new_score,
                    "risk_level": get_risk_level(new_score),
                    "reason": merged_reasons,
                    "updated_at": datetime.utcnow().isoformat(),
                    "mitre": mitre or existing.get("mitre"),
                    "threat_intel": threat_intel or existing.get("threat_intel"),
                },
                "$inc": {"count": 1},
            },
        )
        return existing["_id"]

    new_alert = {
        "ip": event_dict.get("ip"),
        "user_id": event_dict.get("user_id"),
        "event_type": event_dict.get("event_type"),
        "category": category,
        "count": 1,
        "first_seen": timestamp,
        "last_seen": timestamp,
        "risk_score": score,
        "risk_level": level,
        "anomaly": True,
        "reason": list(reasons),
        "status": "new",
        "status_note": None,
        "status_updated_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "mitre": mitre,
        "threat_intel": threat_intel,
    }
    result = alerts_collection.insert_one(new_alert)

    webhooks_module.deliver(db, "alert.created", {
        "_id": str(result.inserted_id),
        "ip": new_alert.get("ip"),
        "user_id": new_alert.get("user_id"),
        "event_type": new_alert.get("event_type"),
        "category": new_alert.get("category"),
        "risk_score": new_alert.get("risk_score"),
        "risk_level": new_alert.get("risk_level"),
        "reason": new_alert.get("reason"),
        "first_seen": new_alert.get("first_seen"),
    })

    return result.inserted_id


def _update_entity_risk_for_event(event_dict, score, is_anomaly):
    """
    Feeds this event's score into the running, time-decaying UEBA risk
    for both the source IP and the user_id (if present) -- see
    entity_risk.py for the decay/contribution model. Called from every
    place an event gets scored (/detect, demo seeding, simulated
    fraud/network streams) so the Top Risk Entities panel reflects all
    traffic, not just live /detect calls.

    Never raises -- risk scoring is a supplementary signal, not something
    that should ever block event ingestion if it fails.
    """
    try:
        ip = event_dict.get("ip")
        if ip:
            update_entity_risk(db, "ip", ip, score, event_dict.get("event_type"),
                                timestamp=event_dict.get("timestamp"), is_anomaly=is_anomaly)
        user_id = event_dict.get("user_id")
        if user_id:
            update_entity_risk(db, "user", user_id, score, event_dict.get("event_type"),
                                timestamp=event_dict.get("timestamp"), is_anomaly=is_anomaly)
    except Exception as e:
        print(f"entity risk update skipped: {e}")


def _compute_alert_priority(alert):
    """
    Combines three explainable signals into one priority score, used to
    sort the Threats tab and populate the Overview "Top Priority Alerts"
    widget:

      - severity (risk_score, 0-100)   -- how bad this looks
      - volume (event count, capped at +20) -- repeated activity from the
        same source is more urgent than a one-off, even at equal severity
      - recency (exponential decay from last_seen, capped at +30,
        halving roughly every 12 hours) -- an alert active five minutes
        ago should usually outrank an equally severe one from three days
        ago that nobody has closed
      - entity risk (capped at +15) -- the source IP's running, time-
        decaying UEBA risk score (see entity_risk.py). This is what lets
        an alert from a source with a documented history of bad behavior
        outrank an equally-severe alert from a source seen for the first
        time, even before this specific alert's own volume/recency catch up.

    Deliberately a transparent weighted formula, not a learned/black-box
    score -- consistent with the survey finding (Q20) that a lack of
    clear, explainable insight is one of the biggest limitations of
    current monitoring tools. An analyst can see exactly why an alert
    ranked where it did.
    """
    risk_score = alert.get("risk_score", 0) or 0
    count = alert.get("count", 1) or 1

    volume_score = min(count * 2, 20)

    recency_score = 0.0
    ts = _parse_ts(alert.get("last_seen"))
    if ts:
        hours_since = max((datetime.utcnow() - ts).total_seconds() / 3600.0, 0)
        recency_score = 30.0 * math.exp(-hours_since / 12.0)

    entity_score = 0.0
    ip = alert.get("ip")
    if ip:
        try:
            entity = get_entity_risk(db, "ip", ip)
            if entity.get("available"):
                entity_score = min(entity["risk_score"] * 0.15, 15.0)
        except Exception:
            entity_score = 0.0

    return round(risk_score + volume_score + recency_score + entity_score, 1)


# =========================
# DEMO DATA AUTO-SEEDING
# =========================
# On a brand-new deployment (e.g. a fresh Render instance / empty MongoDB
# Atlas cluster) all 7 dashboard tabs would otherwise be blank until real
# events start flowing in. This inserts a small, realistic demo dataset
# the FIRST time the app starts against an empty `logs` collection, so the
# live URL is never a wall of "no data yet" empty states.
#
# Safe to leave in permanently: it checks logs_collection.count_documents({})
# and does nothing once there's any data, so it will not duplicate events
# on every restart/redeploy (Render's free tier spins the service down and
# back up often) or interfere once real e-commerce traffic is connected.
def _demo_events():
    """Builds the demo event list with timestamps relative to *now*, so the
    seeded data always looks recent instead of embedding a fixed past date."""
    now = datetime.utcnow()

    def ts(minutes_ago):
        return (now - timedelta(minutes=minutes_ago)).isoformat()

    events = [
        # --- Threats: brute-force burst from one IP, then a port scan ---
        {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
         "timestamp": ts(58), "role": "user"},
        {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
         "timestamp": ts(57), "role": "user"},
        {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
         "timestamp": ts(56), "role": "user"},
        {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
         "timestamp": ts(55), "role": "user"},
        {"ip": "203.0.113.14", "event_type": "failed_login", "severity": "high",
         "timestamp": ts(54), "role": "user"},  # 5th trips the threshold rule
        {"ip": "198.51.100.23", "event_type": "port_scan", "severity": "high",
         "timestamp": ts(50), "role": "user"},

        # --- Fraud: a spread of transaction sizes, one clearly high-risk ---
        {"ip": "192.0.2.55", "event_type": "payment", "severity": "medium",
         "timestamp": ts(45), "user_id": "cust_2210", "amount": 1899.99, "role": "user"},
        {"ip": "192.0.2.71", "event_type": "transaction", "severity": "low",
         "timestamp": ts(40), "user_id": "cust_5521", "amount": 45.50, "role": "user"},
        {"ip": "192.0.2.90", "event_type": "refund", "severity": "medium",
         "timestamp": ts(35), "user_id": "cust_3390", "amount": 2200.00, "role": "user"},
        {"ip": "192.0.2.14", "event_type": "chargeback", "severity": "high",
         "timestamp": ts(30), "user_id": "cust_8871", "amount": 5400.00, "role": "user"},

        # --- User Behavior: one customer (cust_1120) gets a full session of
        # 6 events so the LSTM Behavioral Risk panel has enough history
        # (min_events=5) to actually score them on the User Behavior tab. ---
        {"ip": "10.0.0.5", "event_type": "login", "severity": "low",
         "timestamp": ts(29), "user_id": "cust_1120", "role": "user"},
        {"ip": "10.0.0.5", "event_type": "browse", "severity": "low",
         "timestamp": ts(27), "user_id": "cust_1120", "role": "user"},
        {"ip": "10.0.0.5", "event_type": "add_to_cart", "severity": "low",
         "timestamp": ts(25), "user_id": "cust_1120", "role": "user"},
        {"ip": "10.0.0.5", "event_type": "checkout", "severity": "low",
         "timestamp": ts(23), "user_id": "cust_1120", "role": "user"},
        {"ip": "10.0.0.5", "event_type": "payment", "severity": "medium",
         "timestamp": ts(22), "user_id": "cust_1120", "amount": 129.00, "role": "user"},
        {"ip": "10.0.0.5", "event_type": "logout", "severity": "low",
         "timestamp": ts(20), "user_id": "cust_1120", "role": "user"},

        # A couple of standalone behavior events for other users, for the
        # plain events table.
        {"ip": "10.0.0.6", "event_type": "profile_update", "severity": "low",
         "timestamp": ts(18), "user_id": "cust_1121", "role": "user"},
        {"ip": "10.0.0.7", "event_type": "password_change", "severity": "medium",
         "timestamp": ts(15), "user_id": "cust_1122", "role": "user"},

        # --- Admin Activity ---
        {"ip": "172.16.0.2", "event_type": "admin_login", "severity": "medium",
         "timestamp": ts(12), "user_id": "admin_greg", "role": "admin"},
        {"ip": "172.16.0.2", "event_type": "config_change", "severity": "high",
         "timestamp": ts(11), "user_id": "admin_greg", "role": "admin"},
        {"ip": "172.16.0.3", "event_type": "role_change", "severity": "high",
         "timestamp": ts(9), "user_id": "admin_sara", "role": "admin"},
        {"ip": "172.16.0.4", "event_type": "user_delete", "severity": "critical",
         "timestamp": ts(7), "user_id": "admin_sara", "role": "admin"},
    ]
    return events


def seed_demo_data():
    if logs_collection.count_documents({}) > 0:
        return  # already has data (real traffic or a previous seed) -- leave it alone

    print("logs collection is empty -- seeding demo dataset...")
    inserted = 0
    for raw in _demo_events():
        try:
            event = LogEvent(**raw)
            score, reasons, category = detector.analyze(event)
            level = get_risk_level(score)
            is_anomaly = score >= 50

            mitre = get_mitre_mapping(event.event_type, category)
            intel = enrich_ip(event.ip)

            logs_collection.insert_one({
                **event.dict(),
                "category": category,
                "risk_score": score,
                "risk_level": level,
                "anomaly": is_anomaly,
                "reason": reasons,
                "mitre": mitre,
                "threat_intel": intel,
            })
            _upsert_alert(event.dict(), score, level, is_anomaly, reasons, category,
                          mitre=mitre, threat_intel=intel)
            _update_entity_risk_for_event(event.dict(), score, is_anomaly)
            inserted += 1
        except Exception as e:
            print(f"  skipped one demo event ({raw.get('event_type')}): {e}")

    print(f"Seeded {inserted} demo events.")


@app.on_event("startup")
def _on_startup():
    seed_demo_data()



# =========================
# API KEY PROTECTION (for write/ingestion endpoints)
# =========================
# Set an environment variable before starting uvicorn:
#   Windows cmd:        set SIEM_API_KEY=your-secret-key-here
#   Windows PowerShell:  $env:SIEM_API_KEY="your-secret-key-here"
#
# Callers must then include this header on protected requests:
#   X-API-Key: your-secret-key-here
#
# If SIEM_API_KEY is not set, protected endpoints are left open — this
# keeps local development/testing frictionless, but means you MUST set
# a real key before deploying publicly (e.g. on Render).
_API_KEY = os.environ.get("SIEM_API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if _API_KEY is None:
        return  # no key configured — open for local dev
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# =========================
# HYBRID RISK FUSION — weighted log-odds evidence fusion
# =========================
# Replaces a flat 0.5*rule + 0.5*lstm average with fusion that first
# converts each component into a calibrated probability (using each
# model's own validation statistics), then combines them as independent
# evidence in log-odds space -- the standard "naive Bayes" evidence-fusion
# idea, applied here without needing labeled hybrid-anomaly examples
# (which don't exist across your two datasets -- email.csv and
# creditcard.csv have no shared label space, and live SIEM events have no
# ground truth at all).
with open(os.path.join(os.path.dirname(__file__), "lstm_threshold.json")) as _f:
    _LSTM_META = json.load(_f)

# Rule-based calibration: centered on the "High" boundary from
# get_risk_level() (utils.py) -- the point where the rule engine itself
# calls an event high-risk. Scale chosen so a Critical-level score (85)
# maps to p ≈ 0.9: p=0.5 at High, climbing steeply toward Critical.
RULE_LOGIT_CENTER = 60.0
RULE_LOGIT_SCALE = 12.0

# LSTM calibration: centered on the model's own anomaly threshold
# (lstm_threshold.json), scaled by the validation-set reconstruction
# error's standard deviation -- p=0.5 exactly at the threshold, moving by
# one "typical validation deviation" per unit of z.
LSTM_ERROR_CENTER = _LSTM_META["threshold"]
LSTM_ERROR_SCALE = _LSTM_META["val_reconstruction_error"]["std"] or 1e-6

# Fusion weights. Equal by default. To retune: score the CERT sequences
# in lstm_top_anomalies.csv (known highest-error sequences) alongside
# their rule-based scores, and grid-search these for the split that best
# separates the flagged top-N from the rest.
FUSION_WEIGHT_RULE = 0.5
FUSION_WEIGHT_LSTM = 0.5


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _rule_probability(rule_score):
    """Rule-based 0-100 score -> calibrated probability."""
    z = (rule_score - RULE_LOGIT_CENTER) / RULE_LOGIT_SCALE
    return _sigmoid(z)


def _lstm_probability(reconstruction_error):
    """LSTM reconstruction error -> calibrated probability, using the
    model's own validation threshold and spread."""
    z = (reconstruction_error - LSTM_ERROR_CENTER) / LSTM_ERROR_SCALE
    return _sigmoid(z)


def fuse_hybrid_score(rule_score, lstm_result):
    """
    Combines the rule-based score and the LSTM's result into one unified
    0-100 score via weighted log-odds fusion.

    rule_score: 0-100 float from the rule engine (max over recent events)
    lstm_result: dict from score_user_sequence() -- may have
                 available=False if there isn't enough sequence history.

    Returns: (unified_score: float, detail: dict) -- detail exposes the
             per-component probabilities for transparency/debugging, and
             is handy for worked examples in the Evaluation Framework
             section (e.g. "event X: p_rule=0.71, p_lstm=0.44 -> p=0.66").
    """
    p_rule = _rule_probability(rule_score)

    if lstm_result.get("available"):
        p_lstm = _lstm_probability(lstm_result["reconstruction_error"])
        combined_logit = (
            FUSION_WEIGHT_RULE * _logit(p_rule)
            + FUSION_WEIGHT_LSTM * _logit(p_lstm)
        )
        unified_p = _sigmoid(combined_logit)
        detail = {"p_rule": round(p_rule, 4), "p_lstm": round(p_lstm, 4), "p_unified": round(unified_p, 4)}
    else:
        # No sequence history yet -- fall back to the rule-based
        # probability alone rather than guessing at a fused number.
        unified_p = p_rule
        detail = {"p_rule": round(p_rule, 4), "p_lstm": None, "p_unified": round(p_rule, 4)}

    return round(unified_p * 100, 1), detail


# Root test
@app.get("/")
def serve_dashboard():
    # Serves dashboard.html directly at the root URL, so visiting the
    # Render URL in a browser shows the actual dashboard instead of a
    # bare JSON message. dashboard.html must sit in the same folder as
    # main.py (the project root) for this relative path to work.
    return FileResponse("dashboard.html")


@app.get("/health")
def health():
    # The old root endpoint, moved here. Useful for quick uptime checks
    # (e.g. from the dashboard's own connection-status indicator) without
    # needing to parse HTML.
    return {"message": "Smart SIEM API Running"}


# =========================
# AUTH ENDPOINTS — see auth.py for the design rationale. These are the
# only analyst-auth endpoints that don't themselves require a valid
# token (login is how you get one; nothing else here is unauthenticated).
# =========================
@app.post("/auth/login")
def login(payload: LoginRequest):
    user = auth.authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    auth.touch_last_login(db, str(user["_id"]))
    token = auth.create_access_token(str(user["_id"]), user["username"], user["role"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": auth.ACCESS_TOKEN_EXPIRE_HOURS,
        "user": {
            "username": user["username"],
            "role": user["role"],
            "display_name": user.get("display_name") or user["username"],
        },
    }


@app.get("/auth/me")
def get_me(user: dict = Depends(auth.get_current_user)):
    return user


# --- Admin-only: manage analyst accounts, from the dashboard's Settings
# tab. There's no self-registration flow on purpose -- accounts are
# provisioned by an admin (or via create_admin_user.py for the very
# first one), the way a real SOC controls who gets console access. ---
@app.post("/auth/users", dependencies=[Depends(auth.require_role("admin"))])
def create_user_endpoint(payload: UserCreateRequest):
    try:
        user_id = auth.create_user(
            db, payload.username, payload.password,
            role=payload.role, display_name=payload.display_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "User created", "user_id": user_id}


@app.get("/auth/users", dependencies=[Depends(auth.require_role("admin"))])
def list_users_endpoint():
    return {"results": auth.list_users(db)}


@app.patch("/auth/users/{user_id}/active", dependencies=[Depends(auth.require_role("admin"))])
def set_user_active_endpoint(user_id: str, payload: UserActiveUpdate):
    try:
        found = auth.set_user_active(db, user_id, payload.active)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Updated", "active": payload.active}


# =========================
# CERT DATA INGESTION
# =========================
@app.post("/ingest-cert", dependencies=[Depends(require_api_key)])
def ingest_cert_data():

    logs = get_logs()
    collection.insert_many(logs)

    return {
        "message": "CERT data inserted successfully",
        "count": len(logs)
    }


# =========================
# ADD SINGLE LOG
# =========================
@app.post("/logs", dependencies=[Depends(require_api_key)])
def create_log(log: LogData):

    normalized = normalize_log(log)
    result = collection.insert_one(normalized)

    return {
        "message": "Log stored",
        "id": str(result.inserted_id)
    }


# =========================
# GET LOGS (paginated + filterable) -> powers the "Logs" tab
# =========================
@app.get("/logs", dependencies=[Depends(auth.get_current_user)])
def get_logs_api(
    page: int = 1,
    page_size: int = 25,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    ip: Optional[str] = None,
    search: Optional[str] = None,
):
    query = {}

    if event_type:
        query["event_type"] = event_type
    if severity:
        query["severity"] = severity.lower()
    if ip:
        query["ip"] = ip
    if search:
        # Loose text match across a couple of common fields
        query["$or"] = [
            {"event_type": {"$regex": search, "$options": "i"}},
            {"ip": {"$regex": search, "$options": "i"}},
            {"user_id": {"$regex": search, "$options": "i"}},
        ]

    skip = max(page - 1, 0) * page_size
    total = logs_collection.count_documents(query)

    cursor = logs_collection.find(query).sort("_id", -1).skip(skip).limit(page_size)

    logs = []
    for log in cursor:
        log["_id"] = str(log["_id"])
        logs.append(log)

    return {
        "results": logs,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size if page_size else 0,
    }


# =========================
# AI DETECTION
# =========================
@app.post("/detect", dependencies=[Depends(require_api_key)])
def detect(event: LogEvent):

    score, reasons, category = detector.analyze(event)
    level = get_risk_level(score)
    is_anomaly = score >= 50

    # --- Enrichment: MITRE ATT&CK technique mapping + IP threat intel ---
    # (see mitre_attack.py / threat_intel.py). Both are None-safe -- a
    # failed lookup or an unmapped event type never blocks ingestion.
    mitre = get_mitre_mapping(event.event_type, category)
    intel = enrich_ip(event.ip)

    logs_collection.insert_one({
        **event.dict(),
        "category": category,
        "risk_score": score,
        "risk_level": level,
        "anomaly": is_anomaly,
        "reason": reasons,
        "mitre": mitre,
        "threat_intel": intel,
    })

    alert_id = _upsert_alert(
        event.dict(), score, level, is_anomaly, reasons, category,
        mitre=mitre, threat_intel=intel,
    )
    _update_entity_risk_for_event(event.dict(), score, is_anomaly)

    return {
        "anomaly": is_anomaly,
        "risk_score": score,
        "risk_level": level,
        "category": category,
        "reason": reasons,
        "mitre": mitre,
        "threat_intel": intel,
        "alert_id": str(alert_id) if alert_id else None,
    }


# =========================
# ALERT TRIAGE — update an alert's status (New / Investigating / Resolved
# / False Positive). Called directly by the dashboard's Threats tab.
#
# Behind auth.get_current_user (not require_api_key): /detect and /logs
# POST use X-API-Key because they accept external ingestion traffic (e.g.
# a simulated e-commerce site); this endpoint is a human analyst action
# from inside the dashboard, so it uses the analyst JWT session instead
# (see auth.py). Any authenticated analyst or admin may triage alerts --
# this isn't admin-gated, since triage is core day-to-day SOC work.
# =========================
@app.patch("/alerts/{alert_id}/status")
def update_alert_status(alert_id: str, update: AlertStatusUpdate,
                         user: dict = Depends(auth.get_current_user)):
    if update.status not in ALERT_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {sorted(ALERT_STATUSES)}",
        )
    try:
        obj_id = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_id")

    previous_alert = alerts_collection.find_one({"_id": obj_id})
    if previous_alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    previous_status = previous_alert.get("status")

    result = alerts_collection.update_one(
        {"_id": obj_id},
        {"$set": {
            "status": update.status,
            "status_note": update.note,
            "status_updated_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert = alerts_collection.find_one({"_id": obj_id})
    alert["_id"] = str(alert["_id"])

    webhooks_module.deliver(db, "alert.status_changed", {
        "_id": alert["_id"],
        "ip": alert.get("ip"),
        "event_type": alert.get("event_type"),
        "risk_level": alert.get("risk_level"),
        "previous_status": previous_status,
        "status": update.status,
        "note": update.note,
        "changed_by": user["username"],
    })

    return {"message": "Status updated", "alert": alert}


# =========================
# CASE MANAGEMENT ENDPOINTS — all behind auth.get_current_user (any
# authenticated analyst or admin; case work isn't admin-gated). The
# real logged-in username is now threaded through as `author` on every
# write, replacing the previous hardcoded author="analyst" default in
# cases.py -- so the investigation timeline actually records WHO did
# what, which is the entire point of an audit trail.
# =========================
@app.post("/cases")
def create_case_endpoint(payload: CaseCreate, user: dict = Depends(auth.get_current_user)):
    try:
        case = cases_module.create_case(
            db, payload.title, alert_ids=payload.alert_ids, tags=payload.tags,
            author=user["username"],
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_id in alert_ids")
    return case


@app.get("/cases")
def list_cases_endpoint(status: Optional[str] = None, assigned_to: Optional[str] = None, limit: int = 100,
                         user: dict = Depends(auth.get_current_user)):
    if status and status not in cases_module.CASE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {sorted(cases_module.CASE_STATUSES)}",
        )
    results = cases_module.list_cases(db, status=status, assigned_to=assigned_to, limit=limit)
    return {"results": results, "count": len(results)}


@app.get("/cases/{case_id}")
def get_case_endpoint(case_id: str, user: dict = Depends(auth.get_current_user)):
    try:
        case = cases_module.get_case(db, case_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id")
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.patch("/cases/{case_id}")
def update_case_endpoint(case_id: str, payload: CaseUpdate, user: dict = Depends(auth.get_current_user)):
    if payload.status and payload.status not in cases_module.CASE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of {sorted(cases_module.CASE_STATUSES)}",
        )
    try:
        case = cases_module.update_case(db, case_id, payload.dict(exclude_unset=True), author=user["username"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id")
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.post("/cases/{case_id}/link-alert")
def link_alert_endpoint(case_id: str, payload: CaseAlertLink, user: dict = Depends(auth.get_current_user)):
    try:
        result = cases_module.link_alert(db, case_id, payload.alert_id, author=user["username"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id or alert_id")
    if not result:
        raise HTTPException(status_code=404, detail="Case or alert not found")
    return result


@app.post("/cases/{case_id}/unlink-alert")
def unlink_alert_endpoint(case_id: str, payload: CaseAlertLink, user: dict = Depends(auth.get_current_user)):
    try:
        result = cases_module.unlink_alert(db, case_id, payload.alert_id, author=user["username"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id or alert_id")
    if not result:
        raise HTTPException(status_code=404, detail="Case not found")
    return result


@app.post("/cases/{case_id}/notes")
def add_case_note_endpoint(case_id: str, payload: CaseNoteCreate, user: dict = Depends(auth.get_current_user)):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Note content cannot be empty")
    try:
        result = cases_module.add_note(db, case_id, payload.content.strip(), author=user["username"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id")
    if not result:
        raise HTTPException(status_code=404, detail="Case not found")
    return result


@app.get("/cases/{case_id}/suggested-alerts")
def suggested_alerts_endpoint(case_id: str, limit: int = 10, user: dict = Depends(auth.get_current_user)):
    try:
        results = cases_module.suggest_related_alerts(db, case_id, limit=limit)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid case_id")
    return {"results": results, "count": len(results)}


# =========================
# WEBHOOK MANAGEMENT ENDPOINTS — create/view/test open to any
# authenticated analyst (configuring your own team's Slack notification
# is routine setup work); DELETE is admin-only, since removing an
# integration everyone relies on shouldn't be a single analyst's call.
# =========================
@app.post("/webhooks", dependencies=[Depends(auth.get_current_user)])
def create_webhook_endpoint(payload: WebhookCreate):
    try:
        wh = webhooks_module.register_webhook(
            db, payload.url, payload.events, secret=payload.secret,
            description=payload.description, active=payload.active,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return wh


@app.get("/webhooks", dependencies=[Depends(auth.get_current_user)])
def list_webhooks_endpoint(active_only: bool = False):
    results = webhooks_module.list_webhooks(db, active_only=active_only)
    return {"results": results, "count": len(results), "available_events": sorted(webhooks_module.VALID_EVENTS)}


@app.get("/webhooks/{webhook_id}", dependencies=[Depends(auth.get_current_user)])
def get_webhook_endpoint(webhook_id: str):
    try:
        wh = webhooks_module.get_webhook(db, webhook_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook_id")
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return wh


@app.patch("/webhooks/{webhook_id}", dependencies=[Depends(auth.get_current_user)])
def update_webhook_endpoint(webhook_id: str, payload: WebhookUpdate):
    try:
        wh = webhooks_module.update_webhook(db, webhook_id, payload.dict(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook_id")
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return wh


@app.delete("/webhooks/{webhook_id}", dependencies=[Depends(auth.require_role("admin"))])
def delete_webhook_endpoint(webhook_id: str):
    try:
        deleted = webhooks_module.delete_webhook(db, webhook_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook_id")
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"message": "Webhook deleted"}


@app.post("/webhooks/{webhook_id}/test", dependencies=[Depends(auth.get_current_user)])
def test_webhook_endpoint(webhook_id: str):
    try:
        result = webhooks_module.test_webhook(db, webhook_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook_id")
    if result is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return result


@app.get("/webhooks/{webhook_id}/deliveries", dependencies=[Depends(auth.get_current_user)])
def webhook_deliveries_endpoint(webhook_id: str, limit: int = 20):
    try:
        results = webhooks_module.get_deliveries(db, webhook_id, limit=limit)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook_id")
    return {"results": results, "count": len(results)}


# =========================
# OVERVIEW TAB
# =========================
@app.get("/overview", dependencies=[Depends(auth.get_current_user)])
def overview():

    total = logs_collection.count_documents({})
    critical = logs_collection.count_documents({"risk_level": "Critical"})
    high = logs_collection.count_documents({"risk_level": "High"})
    anomalies = logs_collection.count_documents({"anomaly": True})

    by_category = {}
    for cat in ["threat", "fraud", "user_behavior", "admin_activity"]:
        by_category[cat] = logs_collection.count_documents({"category": cat})

    return {
        "total_events": total,
        "critical_events": critical,
        "high_events": high,
        "anomalies": anomalies,
        "events_by_category": by_category,
    }


def _parse_ts(value):
    """Best-effort parser for the timestamp strings clients send us --
    they're plain strings on LogEvent, not a Mongo Date type, so we parse
    on read rather than relying on Mongo to sort/bucket them for us."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


# =========================
# OVERVIEW: EVENTS-OVER-TIME TREND (for the Overview trend chart)
# =========================
@app.get("/overview/timeline", dependencies=[Depends(auth.get_current_user)])
def overview_timeline(hours: int = 24, buckets: int = 24, scan_limit: int = 3000):
    """
    Buckets recent events into evenly-sized time windows so the Overview
    tab can plot volume over time -- the thing a real SOC dashboard is
    fundamentally built around -- instead of only a point-in-time snapshot.
    """
    now = datetime.utcnow()
    window = timedelta(hours=hours)
    window_start = now - window
    bucket_size = window / buckets

    labels = []
    for i in range(buckets):
        b_start = window_start + bucket_size * i
        labels.append(b_start.strftime("%H:%M") if hours <= 48 else b_start.strftime("%m/%d"))

    total_counts = [0] * buckets
    anomaly_counts = [0] * buckets

    cursor = logs_collection.find(
        {}, {"timestamp": 1, "anomaly": 1}
    ).sort("_id", -1).limit(scan_limit)

    for doc in cursor:
        ts = _parse_ts(doc.get("timestamp"))
        if ts is None or ts < window_start or ts > now:
            continue
        idx = int((ts - window_start) / bucket_size)
        idx = min(max(idx, 0), buckets - 1)
        total_counts[idx] += 1
        if doc.get("anomaly"):
            anomaly_counts[idx] += 1

    return {
        "labels": labels,
        "total": total_counts,
        "anomalies": anomaly_counts,
        "hours": hours,
    }


# =========================
# OVERVIEW: TOP OFFENDING IPs / RISKIEST USERS
# =========================
@app.get("/overview/top-entities", dependencies=[Depends(auth.get_current_user)])
def overview_top_entities(scan_limit: int = 500, top_n: int = 5):
    """
    Surfaces the "top offenders" panel a SOC analyst checks first: which
    IPs and users generated the most events recently, and the highest
    risk seen from each, computed over the most recent `scan_limit` events.
    """
    cursor = logs_collection.find(
        {}, {"ip": 1, "user_id": 1, "risk_score": 1, "risk_level": 1}
    ).sort("_id", -1).limit(scan_limit)

    ip_stats = {}
    user_stats = {}

    def _bump(stats, key, field_name, doc):
        entry = stats.setdefault(key, {field_name: key, "count": 0, "max_risk": 0, "max_level": "Low"})
        entry["count"] += 1
        score = doc.get("risk_score") or 0
        if score >= entry["max_risk"]:
            entry["max_risk"] = score
            entry["max_level"] = doc.get("risk_level") or entry["max_level"]

    for doc in cursor:
        if doc.get("ip"):
            _bump(ip_stats, doc["ip"], "ip", doc)
        if doc.get("user_id"):
            _bump(user_stats, doc["user_id"], "user_id", doc)

    top_ips = sorted(ip_stats.values(), key=lambda x: (-x["count"], -x["max_risk"]))[:top_n]
    top_users = sorted(user_stats.values(), key=lambda x: (-x["count"], -x["max_risk"]))[:top_n]

    return {"top_ips": top_ips, "top_users": top_users}


# =========================
# OVERVIEW: THREAT MAP — geolocated points for offending external IPs
# =========================
@app.get("/overview/threat-map", dependencies=[Depends(auth.get_current_user)])
def overview_threat_map(scan_limit: int = 300, top_n: int = 15):
    """
    Geolocated points for the top offending external IPs -- powers a
    world-map style panel on the Overview tab. Reads threat_intel already
    stored on each log document (see /detect and seed_demo_data) rather
    than re-enriching on request, so this endpoint is cheap to call.

    Private IPs and IPs whose geolocation failed to resolve (e.g. the
    RFC 5737 documentation ranges used by the demo/simulated data -- see
    threat_intel.py's module docstring) are naturally excluded, since
    they never got lat/lon populated in the first place.
    """
    cursor = logs_collection.find(
        {"threat_intel.is_private": False, "threat_intel.lat": {"$ne": None}},
        {"ip": 1, "threat_intel": 1, "risk_level": 1},
    ).sort("_id", -1).limit(scan_limit)

    points = {}
    for doc in cursor:
        intel = doc.get("threat_intel") or {}
        ip = doc.get("ip")
        if not ip or intel.get("lat") is None:
            continue
        entry = points.setdefault(ip, {
            "ip": ip, "lat": intel.get("lat"), "lon": intel.get("lon"),
            "country": intel.get("country"), "city": intel.get("city"),
            "reputation": intel.get("reputation"), "count": 0,
        })
        entry["count"] += 1

    top_points = sorted(points.values(), key=lambda p: -p["count"])[:top_n]
    return {"points": top_points}


# =========================
# THREATS TAB
# =========================
def _correlate_incidents(events, window_minutes=10):
    """
    DEPRECATED / UNUSED as of the alert-triage refactor -- /threats now
    reads from the persistent alerts_collection instead (see
    _upsert_alert above), so triage status survives across requests.
    Kept here for reference on the original on-the-fly grouping approach.

    Groups repeated events sharing the same IP + event_type into a single
    incident when they occur within `window_minutes` of each other -- e.g.
    5 failed_login attempts become one "5x failed_login" incident, the way
    a real SIEM correlates raw events instead of listing each one alone.
    Events not sharing an IP+event_type with a recent neighbor stay as
    their own single-event incident, so nothing is dropped.
    """
    window = timedelta(minutes=window_minutes)
    ordered = sorted(events, key=lambda e: _parse_ts(e.get("timestamp")) or datetime.min)

    open_incidents = {}   # (ip, event_type) -> incident dict, while still within the window
    incidents = []

    for e in ordered:
        ts = _parse_ts(e.get("timestamp"))
        key = (e.get("ip"), e.get("event_type"))
        inc = open_incidents.get(key)

        if inc and ts and inc["_last_ts"] and (ts - inc["_last_ts"]) <= window:
            inc["count"] += 1
            inc["last_seen"] = e.get("timestamp")
            inc["_last_ts"] = ts
            score = e.get("risk_score") or 0
            if score >= inc["risk_score"]:
                inc["risk_score"] = score
                inc["risk_level"] = e.get("risk_level")
            if e.get("anomaly"):
                inc["anomaly"] = True
            for r in (e.get("reason") or []):
                if r not in inc["reason"]:
                    inc["reason"].append(r)
        else:
            inc = {
                "ip": e.get("ip"),
                "user_id": e.get("user_id"),
                "event_type": e.get("event_type"),
                "count": 1,
                "first_seen": e.get("timestamp"),
                "last_seen": e.get("timestamp"),
                "_last_ts": ts,
                "risk_score": e.get("risk_score") or 0,
                "risk_level": e.get("risk_level"),
                "anomaly": bool(e.get("anomaly")),
                "reason": list(e.get("reason") or []),
            }
            open_incidents[key] = inc
            incidents.append(inc)

    for inc in incidents:
        inc.pop("_last_ts", None)

    # Most recent / most severe incidents first
    incidents.sort(key=lambda i: (i["last_seen"] or "", i["risk_score"]), reverse=True)
    return incidents


@app.get("/threats", dependencies=[Depends(auth.get_current_user)])
def threats(limit: int = 100, anomaly_only: bool = True, group_incidents: bool = True,
            correlation_window_minutes: int = 10, scan_limit: int = 500,
            status: Optional[str] = None, sort: str = "priority"):
    if not group_incidents:
        # Raw ungrouped view: every individual threat-category log event,
        # unchanged from before this refactor.
        query = {"category": "threat"}
        if anomaly_only:
            query["anomaly"] = True
        cursor = logs_collection.find(query).sort("_id", -1).limit(limit)
        results = []
        for log in cursor:
            log["_id"] = str(log["_id"])
            results.append(log)
        return {"results": results, "count": len(results), "grouped": False}

    # Grouped/triageable view: reads from the persistent alerts collection
    # (see _upsert_alert) instead of recomputing correlation from raw logs
    # on every request. Alerts are only ever created for anomalous events,
    # so anomaly_only is effectively always satisfied here -- it's kept as
    # a parameter for API compatibility with the raw mode above.
    query = {"category": "threat"}
    if status:
        if status == "open":
            # Convenience shortcut for "still needs attention" -- used by
            # the Overview tab's Top Priority Alerts widget.
            query["status"] = {"$in": ["new", "investigating"]}
        elif status not in ALERT_STATUSES:
            raise HTTPException(status_code=400, detail=f"status must be one of {sorted(ALERT_STATUSES)} or 'open'")
        else:
            query["status"] = status

    # Fetch a wider candidate pool than `limit` so priority sorting (a
    # Python-side computation, not something Mongo can sort on directly)
    # has enough alerts to rank before truncating to the requested limit.
    cursor = alerts_collection.find(query).limit(scan_limit)
    candidates = list(cursor)
    for a in candidates:
        a["priority_score"] = _compute_alert_priority(a)

    if sort == "recency":
        candidates.sort(key=lambda a: a.get("last_seen") or "", reverse=True)
    else:
        candidates.sort(key=lambda a: a["priority_score"], reverse=True)

    results = candidates[:limit]
    for a in results:
        a["_id"] = str(a["_id"])

    return {
        "results": results,
        "count": len(results),
        "grouped": True,
        "sort": sort,
    }


# =========================
# FRAUD TAB
# =========================
@app.get("/fraud", dependencies=[Depends(auth.get_current_user)])
def fraud(limit: int = 100, min_amount: Optional[float] = None):
    query = {"category": "fraud"}
    if min_amount is not None:
        query["amount"] = {"$gte": min_amount}

    cursor = logs_collection.find(query).sort("_id", -1).limit(limit)
    results = []
    for log in cursor:
        log["_id"] = str(log["_id"])
        results.append(log)

    total_flagged_amount = sum(r.get("amount") or 0 for r in results)

    return {
        "results": results,
        "count": len(results),
        "total_flagged_amount": total_flagged_amount,
    }


# =========================
# USER BEHAVIOR TAB
# =========================
@app.get("/user-behavior", dependencies=[Depends(auth.get_current_user)])
def user_behavior(limit: int = 100, user_id: Optional[str] = None):
    query = {"category": "user_behavior"}
    if user_id:
        query["user_id"] = user_id

    cursor = logs_collection.find(query).sort("_id", -1).limit(limit)
    results = []
    for log in cursor:
        log["_id"] = str(log["_id"])
        results.append(log)

    return {"results": results, "count": len(results)}


# =========================
# LSTM BEHAVIORAL RISK (per user) — sequence-based anomaly score,
# the "LSTM" half of the Hybrid Autoencoder + LSTM model. Complements
# the rule-based /user-behavior results above with a reconstruction-error
# based score computed over that user's recent event sequence.
# =========================
@app.get("/user-behavior/{user_id}/lstm-risk", dependencies=[Depends(auth.get_current_user)])
def user_behavior_lstm_risk(user_id: str):
    events = list(
        logs_collection.find({"user_id": user_id}).sort("timestamp", 1).limit(15)
    )
    return score_user_sequence(events)


# =========================
# HYBRID RISK (per user) — Objective 4/hypothesis from the proposal:
# "A hybrid approach integrating Autoencoders, LSTM... is expected to
# perform better... than relying on a single detection technique."
# Fuses the rule-based engine's recent risk for this user with the LSTM's
# sequence-based behavioral score into one unified score + decision,
# via weighted log-odds fusion (see fuse_hybrid_score() above).
#
# The fraud Autoencoder is deliberately NOT included in this live fusion:
# it requires the PCA-anonymized V1-V28 feature vector from the training
# dataset, which live application events don't produce (see the docstring
# in fraud-inference.py). Including it here would mean silently feeding
# it zeros/garbage for those 28 features, which would be dishonest rather
# than a real hybrid signal -- so it stays evaluated offline instead,
# with real metrics in metrics_summary.json.
# =========================
@app.get("/user-behavior/{user_id}/hybrid-risk", dependencies=[Depends(auth.get_current_user)])
def user_behavior_hybrid_risk(user_id: str, recent_limit: int = 20):
    recent_events = list(
        logs_collection.find({"user_id": user_id}).sort("_id", -1).limit(recent_limit)
    )
    if not recent_events:
        return {"available": False, "reason": f"No recent events found for user '{user_id}'."}

    rule_max_score = max((e.get("risk_score") or 0) for e in recent_events)
    rule_anomaly = any(bool(e.get("anomaly")) for e in recent_events)
    rule_level = get_risk_level(rule_max_score)

    seq_events = list(
        logs_collection.find({"user_id": user_id}).sort("timestamp", 1).limit(15)
    )
    lstm_result = score_user_sequence(seq_events)

    lstm_score = lstm_result.get("behavior_score")
    lstm_anomaly = lstm_result.get("is_anomaly", False)

    unified_score, fusion_detail = fuse_hybrid_score(rule_max_score, lstm_result)

    # OR-ensemble decision: either model firing is enough to escalate.
    # Chosen deliberately over requiring both to agree, since a false
    # negative (missed threat) is more costly than a false positive here.
    hybrid_anomaly = rule_anomaly or lstm_anomaly

    return {
        "available": True,
        "user_id": user_id,
        "unified_score": unified_score,
        "unified_level": get_risk_level(unified_score),
        "hybrid_anomaly": hybrid_anomaly,
        "fusion_method": "weighted_log_odds_fusion + OR_ensemble_decision",
        "fusion_detail": fusion_detail,
        "components": {
            "rule_based": {
                "score": rule_max_score,
                "level": rule_level,
                "anomaly": rule_anomaly,
                "events_considered": len(recent_events),
            },
            "lstm_behavioral": {
                "available": lstm_result.get("available", False),
                "score": lstm_score,
                "anomaly": lstm_anomaly,
                "top_features": lstm_result.get("top_features", []),
                "reason": lstm_result.get("reason"),
            },
        },
        "note": (
            "Fuses rule-based risk with LSTM sequence-based behavioral risk "
            "using weighted log-odds fusion: each component is converted to "
            "a calibrated probability using its own validation statistics, "
            "then combined as independent evidence in log-odds space. The "
            "fraud Autoencoder is evaluated offline only -- see "
            "fraud-inference.py and metrics_summary.json -- because it needs "
            "the PCA-anonymized feature vector live events don't produce."
        ),
    }


# =========================
# ENTITY RISK (UEBA) — persistent, time-decaying risk per IP/user,
# separate from any single event or alert. See entity_risk.py for the
# decay/contribution model. Powers the Overview "Top Risk Entities" panel
# and per-entity risk-timeline lookups.
# =========================
@app.get("/entity-risk/top", dependencies=[Depends(auth.get_current_user)])
def entity_risk_top(entity_type: Optional[str] = None, limit: int = 10, min_score: float = 0.0):
    if entity_type and entity_type not in ("ip", "user"):
        raise HTTPException(status_code=400, detail="entity_type must be 'ip' or 'user'")
    results = get_top_risk_entities(db, entity_type=entity_type, limit=limit, min_score=min_score)
    return {"results": results, "count": len(results)}


@app.get("/entity-risk/{entity_type}/{entity_id}", dependencies=[Depends(auth.get_current_user)])
def entity_risk_detail(entity_type: str, entity_id: str):
    if entity_type not in ("ip", "user"):
        raise HTTPException(status_code=400, detail="entity_type must be 'ip' or 'user'")
    return get_entity_risk(db, entity_type, entity_id)


# =========================
# ADMIN ACTIVITY TAB
# =========================
@app.get("/admin-activity", dependencies=[Depends(auth.get_current_user)])
def admin_activity(limit: int = 100):
    cursor = logs_collection.find({"category": "admin_activity"}).sort("_id", -1).limit(limit)
    results = []
    for log in cursor:
        log["_id"] = str(log["_id"])
        results.append(log)

    return {"results": results, "count": len(results)}


# =========================
# AI INSIGHTS TAB — free, rule-based analysis (see ai_insights.py)
# =========================
@app.get("/ai-insights", dependencies=[Depends(auth.get_current_user)])
def ai_insights(limit: int = 50):
    recent_events = list(
        logs_collection.find().sort("_id", -1).limit(limit)
    )
    for e in recent_events:
        e["_id"] = str(e["_id"])

    return generate_insights(recent_events)


# =========================
# STATS (kept for backward compatibility with existing prototype)
# =========================
@app.get("/stats", dependencies=[Depends(auth.get_current_user)])
def stats():

    total = logs_collection.count_documents({})
    critical = logs_collection.count_documents({"risk_level": "Critical"})
    anomalies = logs_collection.count_documents({"anomaly": True})

    return {
        "total_events": total,
        "critical_events": critical,
        "anomalies": anomalies
    }
    
@app.post("/fraud/simulate", dependencies=[Depends(require_api_key)])
def simulate_fraud_stream(count: int = 20):
    events = generate_simulated_fraud_events(count=count)
    inserted = 0
    for e in events:
        logs_collection.insert_one(e)
        _upsert_alert(e, e["risk_score"], e["risk_level"], e["anomaly"], e["reason"], "fraud")
        _update_entity_risk_for_event(e, e["risk_score"], e["anomaly"])
        inserted += 1
    return {
        "message": f"Inserted {inserted} simulated fraud events",
        "count": inserted,
        "source": "simulated_fraud_stream",
    }


# =========================
# SIMULATED NETWORK INTRUSION STREAM — replays held-out UNSW-NB15 test
# flows through the real trained Network Intrusion Autoencoder (see
# network_stream.py for why this is the honest way to demo the model
# live, given live SIEM events don't produce UNSW-NB15's 45 flow-level
# features). Inserted as category="threat" so it flows into the same
# Threats tab / alert pipeline as the rule-based detections.
# =========================
@app.post("/threats/simulate-network", dependencies=[Depends(require_api_key)])
def simulate_network_stream(count: int = 20):
    events = generate_simulated_network_events(count=count)
    inserted = 0
    for e in events:
        logs_collection.insert_one(e)
        _upsert_alert(e, e["risk_score"], e["risk_level"], e["anomaly"], e["reason"], "threat")
        _update_entity_risk_for_event(e, e["risk_score"], e["anomaly"])
        inserted += 1
    return {
        "message": f"Inserted {inserted} simulated network intrusion events",
        "count": inserted,
        "source": "simulated_network_stream",
    }
    
@app.get("/entity-risk/predictive", dependencies=[Depends(auth.get_current_user)])
def entity_risk_predictive(entity_type: Optional[str] = None, limit: int = 10,
                            min_velocity: float = 2.0, alert_threshold: float = 50.0):
    if entity_type and entity_type not in ("ip", "user"):
        raise HTTPException(status_code=400, detail="entity_type must be 'ip' or 'user'")
    results = get_escalating_entities(
        db, entity_type=entity_type, limit=limit,
        min_velocity=min_velocity, alert_threshold=alert_threshold,
    )
    return {"results": results, "count": len(results), "alert_threshold": alert_threshold}