"""
Entity Risk Scoring (UEBA) -- persistent, time-decaying risk per IP and
per user, computed across the entity's whole history rather than one
event at a time.

WHY THIS EXISTS: the detector (detector.py) and the LSTM/hybrid fusion
(main.py's fuse_hybrid_score) both score a single EVENT or a single
SEQUENCE. Neither one remembers that "this IP has been mildly suspicious
five separate times over the last three days" -- which is exactly the
kind of slow-burn pattern real User and Entity Behavior Analytics (UEBA)
tools are built to catch, and the exact idea your literature review cites
(Aljumaily, Abd & Majeed, 2025 -- UEBA-based anomaly detection combining
behavioral signal over time, not just point-in-time events).

DESIGN:
  - One document per (entity_type, entity_id) in `entity_risk` collection:
        {
          "entity_type": "ip" | "user",
          "entity_id": str,
          "risk_score": float,        # 0-100, current decayed + updated score
          "event_count": int,         # total events ever contributing
          "last_updated": iso str,
          "history": [ {"ts": iso str, "score": float, "event_type": str,
                        "contribution": float}, ... ]  # capped, most recent last
        }

  - EXPONENTIAL TIME DECAY: on every update, the entity's stored score is
    first decayed based on elapsed time since last_updated, using a
    configurable half-life (default 7 days). This means an entity that
    was risky last month but quiet since naturally cools back down --
    exactly like a real UEBA baseline -- rather than accumulating a
    score forever.

  - CONTRIBUTION: each new event nudges the decayed score toward the
    event's own risk_score (from detector.py / the hybrid fusion),
    weighted by CONTRIBUTION_WEIGHT. This means one bad event doesn't
    instantly max out an entity's risk (avoiding a single false positive
    dominating the running score), but repeated bad events compound.

  - HISTORY: a capped rolling list of recent (timestamp, score) points is
    kept alongside the running score, purely so the dashboard can plot a
    risk trend line for an entity instead of only showing a single
    current number.

This module has no FastAPI dependency -- main.py wires it into /detect
and exposes it via a couple of small endpoints.
"""

from datetime import datetime, timedelta
import math

HALF_LIFE_HOURS = 168.0          # 7 days -- risk halves if nothing new happens
CONTRIBUTION_WEIGHT = 0.35       # how strongly one new event nudges the running score
HISTORY_MAX_POINTS = 50          # cap so documents don't grow unbounded
MAX_SCORE = 100.0


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _decay(score, hours_elapsed):
    """Exponential decay toward 0 with the configured half-life."""
    if hours_elapsed <= 0:
        return score
    factor = 0.5 ** (hours_elapsed / HALF_LIFE_HOURS)
    return score * factor


def update_entity_risk(db, entity_type, entity_id, event_score, event_type,
                        timestamp=None, is_anomaly=False):
    """
    Decays the entity's existing risk score based on elapsed time, then
    blends in the new event's contribution, and appends a history point.

    entity_type: "ip" | "user"
    entity_id:   the IP string or user_id string
    event_score: 0-100 risk_score from detector.py / hybrid fusion for
                 THIS single event
    event_type:  event_type string, stored for context in the history
    timestamp:   event's own ISO timestamp (falls back to now if absent)
    is_anomaly:  whether this single event was flagged anomalous -- boosts
                 the contribution weight slightly, since a confirmed
                 anomaly should move the needle more than a routine event

    Returns the updated document (dict).
    """
    if not entity_id:
        return None

    collection = db.entity_risk
    now = _parse_ts(timestamp) or datetime.utcnow()

    existing = collection.find_one({"entity_type": entity_type, "entity_id": entity_id})

    if existing:
        last_updated = _parse_ts(existing.get("last_updated")) or now
        hours_elapsed = max((now - last_updated).total_seconds() / 3600.0, 0)
        decayed_score = _decay(existing.get("risk_score", 0.0), hours_elapsed)
        history = existing.get("history", [])
        event_count = existing.get("event_count", 0)
    else:
        decayed_score = 0.0
        history = []
        event_count = 0

    weight = CONTRIBUTION_WEIGHT * (1.4 if is_anomaly else 1.0)
    new_score = min(decayed_score + (event_score - decayed_score) * weight, MAX_SCORE)
    new_score = max(new_score, 0.0)

    history.append({
        "ts": now.isoformat(),
        "score": round(new_score, 1),
        "event_type": event_type,
        "contribution": round(event_score, 1),
    })
    history = history[-HISTORY_MAX_POINTS:]

    doc = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "risk_score": round(new_score, 1),
        "event_count": event_count + 1,
        "last_updated": now.isoformat(),
        "history": history,
    }

    collection.update_one(
        {"entity_type": entity_type, "entity_id": entity_id},
        {"$set": doc},
        upsert=True,
    )
    return doc


def get_entity_risk(db, entity_type, entity_id):
    """
    Returns the entity's current risk, with decay applied for display even
    if no new event has arrived since last_updated (so the number shown
    on the dashboard is always "as of right now", not stale).
    """
    doc = db.entity_risk.find_one({"entity_type": entity_type, "entity_id": entity_id})
    if not doc:
        return {"available": False, "reason": f"No risk history for {entity_type} '{entity_id}' yet."}

    last_updated = _parse_ts(doc.get("last_updated"))
    now = datetime.utcnow()
    hours_elapsed = max((now - last_updated).total_seconds() / 3600.0, 0) if last_updated else 0
    current_score = round(_decay(doc.get("risk_score", 0.0), hours_elapsed), 1)

    return {
        "available": True,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "risk_score": current_score,
        "stored_score": doc.get("risk_score", 0.0),
        "event_count": doc.get("event_count", 0),
        "last_updated": doc.get("last_updated"),
        "history": doc.get("history", []),
        "half_life_hours": HALF_LIFE_HOURS,
    }


def get_top_risk_entities(db, entity_type=None, limit=10, min_score=0.0):
    """
    Returns the highest-risk entities, with decay applied for display so
    ranking reflects current risk, not just whatever was last stored.
    Powers the dashboard's "Top Risk Entities" (UEBA) panel.
    """
    query = {}
    if entity_type:
        query["entity_type"] = entity_type

    now = datetime.utcnow()
    results = []
    for doc in db.entity_risk.find(query):
        last_updated = _parse_ts(doc.get("last_updated"))
        hours_elapsed = max((now - last_updated).total_seconds() / 3600.0, 0) if last_updated else 0
        current_score = round(_decay(doc.get("risk_score", 0.0), hours_elapsed), 1)
        if current_score < min_score:
            continue
        results.append({
            "entity_type": doc.get("entity_type"),
            "entity_id": doc.get("entity_id"),
            "risk_score": current_score,
            "event_count": doc.get("event_count", 0),
            "last_updated": doc.get("last_updated"),
            "trend": [h["score"] for h in doc.get("history", [])[-10:]],
        })

    results.sort(key=lambda r: -r["risk_score"])
    return results[:limit]

def _compute_velocity(history, lookback=5):
    """
    Points-per-hour rate of change over the last `lookback` history
    entries. Returns None if there isn't enough history or the time span
    is effectively zero (can't compute a rate).
    """
    if not history or len(history) < 2:
        return None
    recent = history[-lookback:]
    if len(recent) < 2:
        return None
    try:
        t0 = _parse_ts(recent[0]["ts"])
        t1 = _parse_ts(recent[-1]["ts"])
    except Exception:
        return None
    if t0 is None or t1 is None:
        return None
    hours = (t1 - t0).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return (recent[-1]["score"] - recent[0]["score"]) / hours


def get_escalating_entities(db, entity_type=None, limit=10, min_velocity=2.0,
                             alert_threshold=50.0, lookback=5):
    """
    EARLY WARNING / PREDICTIVE SIGNAL -- separate from get_top_risk_entities,
    which ranks by current absolute score. This ranks by TREND: entities
    whose risk is climbing at least `min_velocity` points/hour but hasn't
    crossed `alert_threshold` yet. The idea: a slow burn from Low(15) to
    Medium(35) over a few hours is worth an analyst's attention before it
    ever fires a real alert, the same way a SOC watches for "building
    toward something" not just "already bad".

    HONEST LIMITATION -- state this in the report: eta_hours_to_threshold
    is a NAIVE LINEAR EXTRAPOLATION from the last `lookback` history
    points (constant-rate projection), not a trained time-series/
    forecasting model. Real attacker behavior is often bursty rather than
    linear, so treat this as a rough triage-ordering heuristic ("watch
    this one first"), not a committed prediction of when an attack will
    happen.
    """
    query = {}
    if entity_type:
        query["entity_type"] = entity_type

    now = datetime.utcnow()
    results = []
    for doc in db.entity_risk.find(query):
        velocity = _compute_velocity(doc.get("history", []), lookback=lookback)
        if velocity is None or velocity < min_velocity:
            continue

        last_updated = _parse_ts(doc.get("last_updated"))
        hours_elapsed = max((now - last_updated).total_seconds() / 3600.0, 0) if last_updated else 0
        current_score = round(_decay(doc.get("risk_score", 0.0), hours_elapsed), 1)

        if current_score >= alert_threshold:
            continue  # already at/past the threshold -- that's Top Risk Entities' job, not an early warning anymore

        eta_hours = round((alert_threshold - current_score) / velocity, 1) if velocity > 0 else None

        results.append({
            "entity_type": doc.get("entity_type"),
            "entity_id": doc.get("entity_id"),
            "risk_score": current_score,
            "velocity_per_hour": round(velocity, 2),
            "eta_hours_to_threshold": eta_hours,
            "alert_threshold": alert_threshold,
            "event_count": doc.get("event_count", 0),
            "last_updated": doc.get("last_updated"),
            "trend": [h["score"] for h in doc.get("history", [])[-10:]],
        })

    results.sort(key=lambda r: -r["velocity_per_hour"])
    return results[:limit]