"""
Webhook notifications -- lets external systems (Slack via an incoming
webhook, a ticketing system, a custom listener script, etc.) subscribe
to Smart SIEM events instead of having to poll the API.

DESIGN:
  One document per registered webhook in the `webhooks` collection:
    {
      "_id": ObjectId,
      "url": str,
      "events": [str, ...],       # subset of VALID_EVENTS this webhook wants
      "secret": str | None,       # used to HMAC-sign outgoing payloads
      "description": str,
      "active": bool,
      "created_at": iso str,
      "updated_at": iso str,
    }

  Every delivery attempt (success or failure) is logged to
  `webhook_deliveries` so a misbehaving endpoint is debuggable from the
  dashboard instead of failing silently:
    {
      "_id": ObjectId, "webhook_id": ObjectId, "event_type": str,
      "url": str, "sent_at": iso str, "success": bool,
      "status_code": int | None, "error": str | None,
    }

EVENT TYPES (VALID_EVENTS below):
  alert.created         -- a brand-new alert was raised (not a merge into
                            an existing open alert -- see main.py's
                            _upsert_alert correlation logic)
  alert.status_changed   -- an analyst changed an alert's triage status
  case.created            -- a new investigation case was opened
  case.status_changed     -- a case's status changed (open -> investigating
                            -> contained -> closed)
  case.note_added         -- an analyst added an investigation note to a
                            case (fired only for analyst notes via
                            add_note(), not the automatic "alert linked/
                            unlinked" timeline entries -- keeps this event
                            meaningful instead of noisy)

SECURITY: if a webhook has a secret, every payload is signed with
HMAC-SHA256 over the raw JSON body, sent as the X-SIEM-Signature header
("sha256=<hex>"). The receiving endpoint can recompute the same HMAC to
verify the request actually came from this SIEM and wasn't spoofed --
the standard pattern used by GitHub/Stripe/Slack-style webhooks.

HONEST LIMITATION (state this in the report, same pattern as the other
honesty notes in this codebase): this is a single-attempt, synchronous
delivery -- no retry queue or exponential backoff. A production SIEM
would hand delivery off to a background worker/task queue so a slow or
down endpoint can't add latency to the request that triggered the event
(e.g. /detect). For this project's scale, a short timeout
(DELIVERY_TIMEOUT_SECONDS) and a wrapped try/except keep a failing
webhook from ever blocking or breaking the triggering request -- it just
gets logged as a failed delivery instead of retried.
"""

import hashlib
import hmac
import json
from datetime import datetime

import requests
from bson import ObjectId

VALID_EVENTS = {
    "alert.created",
    "alert.status_changed",
    "case.created",
    "case.status_changed",
    "case.note_added",
}

DELIVERY_TIMEOUT_SECONDS = 4
DELIVERY_HISTORY_CAP = 200  # per-webhook cap on stored delivery log entries


def _now():
    return datetime.utcnow().isoformat()


def _oid(value):
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _redact(doc):
    """Never returns the raw secret over the API once stored -- only
    whether one is set. The secret is still read from the DB internally
    when signing outgoing payloads."""
    d = dict(doc)
    d["_id"] = str(d["_id"])
    d["has_secret"] = bool(d.get("secret"))
    d.pop("secret", None)
    return d


# =========================
# Registration / management
# =========================
def register_webhook(db, url, events, secret=None, description=None, active=True):
    invalid = set(events) - VALID_EVENTS
    if invalid:
        raise ValueError(f"Unknown event type(s): {sorted(invalid)}. Valid: {sorted(VALID_EVENTS)}")
    if not events:
        raise ValueError("At least one event type is required.")

    now = _now()
    doc = {
        "url": url,
        "events": list(events),
        "secret": secret,
        "description": description or "",
        "active": active,
        "created_at": now,
        "updated_at": now,
    }
    result = db.webhooks.insert_one(doc)
    return get_webhook(db, str(result.inserted_id))


def get_webhook(db, webhook_id):
    doc = db.webhooks.find_one({"_id": _oid(webhook_id)})
    return _redact(doc) if doc else None


def list_webhooks(db, active_only=False):
    query = {"active": True} if active_only else {}
    return [_redact(d) for d in db.webhooks.find(query).sort("created_at", -1)]


def update_webhook(db, webhook_id, updates):
    webhook_oid = _oid(webhook_id)
    if not db.webhooks.find_one({"_id": webhook_oid}):
        return None

    allowed = {"url", "events", "secret", "description", "active"}
    set_fields = {k: v for k, v in updates.items() if k in allowed and v is not None}

    if "events" in set_fields:
        invalid = set(set_fields["events"]) - VALID_EVENTS
        if invalid:
            raise ValueError(f"Unknown event type(s): {sorted(invalid)}. Valid: {sorted(VALID_EVENTS)}")

    if not set_fields:
        return get_webhook(db, webhook_id)

    set_fields["updated_at"] = _now()
    db.webhooks.update_one({"_id": webhook_oid}, {"$set": set_fields})
    return get_webhook(db, webhook_id)


def delete_webhook(db, webhook_id):
    webhook_oid = _oid(webhook_id)
    result = db.webhooks.delete_one({"_id": webhook_oid})
    db.webhook_deliveries.delete_many({"webhook_id": webhook_oid})
    return result.deleted_count > 0


# =========================
# Delivery
# =========================
def _log_delivery(db, result):
    db.webhook_deliveries.insert_one(dict(result))

    # Trim oldest entries beyond the cap so this collection doesn't grow
    # unbounded over a long-running deployment.
    count = db.webhook_deliveries.count_documents({"webhook_id": result["webhook_id"]})
    if count > DELIVERY_HISTORY_CAP:
        excess = count - DELIVERY_HISTORY_CAP
        oldest_ids = [
            d["_id"] for d in db.webhook_deliveries
            .find({"webhook_id": result["webhook_id"]}, {"_id": 1})
            .sort("sent_at", 1).limit(excess)
        ]
        if oldest_ids:
            db.webhook_deliveries.delete_many({"_id": {"$in": oldest_ids}})


def _send(db, webhook, event_type, payload):
    """Signs (if a secret is set) and POSTs one event to one webhook,
    logs the attempt, and never raises -- see module docstring."""
    now = _now()
    body = {"event": event_type, "sent_at": now, "data": payload}
    body_json = json.dumps(body, default=str)

    headers = {"Content-Type": "application/json", "X-SIEM-Event": event_type}
    if webhook.get("secret"):
        signature = hmac.new(
            webhook["secret"].encode("utf-8"), body_json.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers["X-SIEM-Signature"] = f"sha256={signature}"

    result = {
        "webhook_id": webhook["_id"],
        "event_type": event_type,
        "url": webhook.get("url"),
        "sent_at": now,
    }
    try:
        res = requests.post(webhook["url"], data=body_json, headers=headers,
                             timeout=DELIVERY_TIMEOUT_SECONDS)
        result["status_code"] = res.status_code
        result["success"] = 200 <= res.status_code < 300
        result["error"] = None if result["success"] else f"Non-2xx response: {res.status_code}"
    except Exception as e:
        result["status_code"] = None
        result["success"] = False
        result["error"] = str(e)

    _log_delivery(db, result)
    return result


def deliver(db, event_type, payload):
    """
    Sends `payload` to every ACTIVE webhook subscribed to `event_type`.
    Called from main.py (alert events) and cases.py (case events) right
    after the triggering write succeeds. Never raises -- a webhook
    delivery failure must never break alert ingestion or case updates.
    Returns the list of per-webhook delivery results (mostly useful for
    tests/debugging; callers in the request path ignore the return value).
    """
    if event_type not in VALID_EVENTS:
        return []
    try:
        subscribers = list(db.webhooks.find({"active": True, "events": event_type}))
    except Exception:
        return []

    results = []
    for webhook in subscribers:
        try:
            results.append(_send(db, webhook, event_type, payload))
        except Exception as e:
            results.append({
                "webhook_id": webhook["_id"], "event_type": event_type,
                "url": webhook.get("url"), "success": False, "error": str(e),
            })
    return results


def test_webhook(db, webhook_id):
    """Sends a synthetic 'webhook.test' ping to one webhook regardless of
    its subscribed events -- lets an analyst verify a URL/secret works
    before relying on it for real alerts."""
    webhook = db.webhooks.find_one({"_id": _oid(webhook_id)})
    if not webhook:
        return None
    payload = {
        "message": "This is a test event from your Smart SIEM webhook integration.",
        "webhook_id": str(webhook["_id"]),
    }
    return _send(db, webhook, "webhook.test", payload)


def get_deliveries(db, webhook_id, limit=20):
    webhook_oid = _oid(webhook_id)
    cursor = db.webhook_deliveries.find({"webhook_id": webhook_oid}).sort("sent_at", -1).limit(limit)
    results = []
    for d in cursor:
        d["_id"] = str(d["_id"])
        d["webhook_id"] = str(d["webhook_id"])
        results.append(d)
    return results