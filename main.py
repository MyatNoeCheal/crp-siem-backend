from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import os

from ai.detector import AnomalyDetector
from models import LogData, LogEvent
from database import get_db
from utils import normalize_log, get_risk_level
from dataset.scripts.clean_email import get_logs
from ai_insights import generate_insights
from lstm_inference import score_user_sequence

import dataset.scripts.clean_email as ce

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
@app.get("/logs")
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
    total = collection.count_documents(query)

    cursor = collection.find(query).sort("_id", -1).skip(skip).limit(page_size)

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

    logs_collection.insert_one({
        **event.dict(),
        "category": category,
        "risk_score": score,
        "risk_level": level,
        "anomaly": is_anomaly
    })

    return {
        "anomaly": is_anomaly,
        "risk_score": score,
        "risk_level": level,
        "category": category,
        "reason": reasons
    }


# =========================
# OVERVIEW TAB
# =========================
@app.get("/overview")
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


# =========================
# THREATS TAB
# =========================
@app.get("/threats")
def threats(limit: int = 100, anomaly_only: bool = True):
    query = {"category": "threat"}
    if anomaly_only:
        query["anomaly"] = True

    cursor = logs_collection.find(query).sort("_id", -1).limit(limit)
    results = []
    for log in cursor:
        log["_id"] = str(log["_id"])
        results.append(log)

    return {"results": results, "count": len(results)}


# =========================
# FRAUD TAB
# =========================
@app.get("/fraud")
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
@app.get("/user-behavior")
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
@app.get("/user-behavior/{user_id}/lstm-risk")
def user_behavior_lstm_risk(user_id: str):
    events = list(
        logs_collection.find({"user_id": user_id}).sort("timestamp", 1).limit(15)
    )
    return score_user_sequence(events)


# =========================
# ADMIN ACTIVITY TAB
# =========================
@app.get("/admin-activity")
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
@app.get("/ai-insights")
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
@app.get("/stats")
def stats():

    total = logs_collection.count_documents({})
    critical = logs_collection.count_documents({"risk_level": "Critical"})
    anomalies = logs_collection.count_documents({"anomaly": True})

    return {
        "total_events": total,
        "critical_events": critical,
        "anomalies": anomalies
    }