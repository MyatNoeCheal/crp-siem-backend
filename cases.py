"""
Case Management -- turns the alert queue (alerts_collection, see main.py's
_upsert_alert) into an actual investigation workflow, the way a real SOC
separates "things the detection engine flagged" (alerts) from "things an
analyst is actively working" (cases).

WHY A SEPARATE LAYER FROM ALERTS: an alert is already deduplicated/
correlated at the single-source level (same IP + event_type + category
within a time window -- see main.py's ALERT_CORRELATION_WINDOW_MINUTES).
A case is one level up: "these 3 alerts across 2 IPs and a compromised
account over 6 hours are actually one incident." Alerts stay the
detection-engine's output; cases are the analyst's own grouping and
investigation record on top of them.

DESIGN:
  One document per case in the `cases` collection:
    {
      "_id": ObjectId,
      "title": str,
      "status": "open" | "investigating" | "contained" | "closed",
      "severity": "Low"|"Medium"|"High"|"Critical",  # derived: max risk_level
                                                        # across linked alerts
      "alert_ids": [ObjectId, ...],
      "entity_ids": [{"type": "ip"|"user", "id": str}, ...],  # derived
      "alert_count": int,          # cached, avoids recomputing on every list
      "assigned_to": str | None,   # free-text analyst name -- no auth system
                                    # exists yet (same reasoning as main.py's
                                    # /alerts/{id}/status not requiring a key)
      "created_at": iso str,
      "updated_at": iso str,
      "tags": [str, ...],
      "timeline": [                # append-only investigation log
        {"ts": iso str, "type": "note"|"status_change"|"alert_linked"|
                                  "alert_unlinked",
         "author": str, "content": str}
      ],
    }

SEVERITY / ENTITY_IDS ARE CACHED, NOT LIVE-COMPUTED ON EVERY READ: they're
recalculated once whenever alerts are linked/unlinked (_recompute_case_
aggregates) and stored back onto the case document. This keeps GET /cases
(the list view) cheap -- it doesn't need to re-fetch every linked alert
for every case just to show a severity badge.

AUTO-SUGGESTION, NOT AUTO-LINKING: suggest_related_alerts() finds OPEN
alerts sharing an IP or user_id with alerts already on the case, and
returns them as candidates for the analyst to review -- it never links
them automatically. Same "analyst stays in the loop" pattern as the
MITRE/threat-intel enrichment in this codebase being informational
rather than auto-actioned; a shared IP is a hint worth surfacing, not
proof the events are related.
"""

from datetime import datetime
from bson import ObjectId

from entity_risk import get_entity_risk
import webhooks

RISK_LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
CASE_STATUSES = {"open", "investigating", "contained", "closed"}


def _now():
    return datetime.utcnow().isoformat()


def _oid(value):
    """Accepts either a string or an ObjectId and always returns an
    ObjectId -- callers pass strings (from API payloads), stored data
    uses real ObjectIds, this normalizes either direction."""
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _recompute_case_aggregates(db, case_oid):
    """
    Recomputes severity, entity_ids, and alert_count from the case's
    currently-linked alerts, and writes them back onto the case document.
    Called after every link/unlink so the cached fields never drift from
    the actual alert_ids list.
    """
    case = db.cases.find_one({"_id": case_oid})
    if not case:
        return

    alert_ids = case.get("alert_ids", [])
    alerts = list(db.alerts.find({"_id": {"$in": alert_ids}})) if alert_ids else []

    if alerts:
        severity = max(
            alerts, key=lambda a: RISK_LEVEL_RANK.get(a.get("risk_level", "Low"), 0)
        ).get("risk_level", "Low")
    else:
        severity = "Low"

    entity_ids = []
    seen = set()
    for a in alerts:
        if a.get("ip") and ("ip", a["ip"]) not in seen:
            seen.add(("ip", a["ip"]))
            entity_ids.append({"type": "ip", "id": a["ip"]})
        if a.get("user_id") and ("user", a["user_id"]) not in seen:
            seen.add(("user", a["user_id"]))
            entity_ids.append({"type": "user", "id": a["user_id"]})

    db.cases.update_one(
        {"_id": case_oid},
        {"$set": {
            "severity": severity,
            "entity_ids": entity_ids,
            "alert_count": len(alerts),
            "updated_at": _now(),
        }},
    )


def create_case(db, title, alert_ids=None, tags=None, author="analyst"):
    """
    Creates a new case, optionally pre-linked to one or more existing
    alerts (e.g. "create a case from this alert" on the Threats tab).
    Returns the full case detail (see get_case) rather than the bare
    insert result, so the caller can render it immediately.
    """
    now = _now()
    doc = {
        "title": title,
        "status": "open",
        "severity": "Low",
        "alert_ids": [],
        "entity_ids": [],
        "alert_count": 0,
        "assigned_to": None,
        "created_at": now,
        "updated_at": now,
        "tags": tags or [],
        "timeline": [{"ts": now, "type": "note", "author": author, "content": "Case created."}],
    }
    result = db.cases.insert_one(doc)
    case_oid = result.inserted_id

    webhooks.deliver(db, "case.created", {
        "_id": str(case_oid), "title": title, "status": "open", "tags": tags or [],
    })

    for alert_id in (alert_ids or []):
        link_alert(db, str(case_oid), alert_id, author=author)

    return get_case(db, str(case_oid))


def list_cases(db, status=None, assigned_to=None, limit=100):
    """
    List view -- deliberately strips the full timeline (can grow large
    over a long investigation) since the case list only needs the
    summary fields. Fetch a single case via get_case() for full detail.
    """
    query = {}
    if status:
        query["status"] = status
    if assigned_to:
        query["assigned_to"] = assigned_to

    cursor = db.cases.find(query).sort("updated_at", -1).limit(limit)
    results = []
    for c in cursor:
        c["_id"] = str(c["_id"])
        c["alert_ids"] = [str(a) for a in c.get("alert_ids", [])]
        c.pop("timeline", None)
        results.append(c)
    return results


def get_case(db, case_id):
    """
    Full case detail: the case document itself, PLUS the resolved alert
    documents for everything in alert_ids (so the dashboard doesn't need
    a second round-trip per alert), PLUS a live entity-risk snapshot
    (via entity_risk.py) for every IP/user touched by this case -- gives
    an analyst the UEBA context for this case's entities without leaving
    the case view.
    """
    case = db.cases.find_one({"_id": _oid(case_id)})
    if not case:
        return None

    alert_docs = list(db.alerts.find({"_id": {"$in": case.get("alert_ids", [])}}))
    for a in alert_docs:
        a["_id"] = str(a["_id"])

    case["_id"] = str(case["_id"])
    case["alert_ids"] = [str(a) for a in case.get("alert_ids", [])]
    case["alerts"] = alert_docs

    entity_risk_snapshot = []
    for e in case.get("entity_ids", []):
        er = get_entity_risk(db, e["type"], e["id"])
        if er.get("available"):
            entity_risk_snapshot.append({
                "type": e["type"], "id": e["id"], "risk_score": er["risk_score"],
            })
    case["entity_risk_snapshot"] = entity_risk_snapshot

    return case


def update_case(db, case_id, updates, author="analyst"):
    """
    Patches title/status/assigned_to/tags. Status changes and assignment
    changes get their own timeline entry automatically -- these are the
    two fields worth an investigation audit trail; title/tag edits don't
    (low-signal, would clutter the timeline).
    """
    case_oid = _oid(case_id)
    existing_case = db.cases.find_one({"_id": case_oid})
    if not existing_case:
        return None

    now = _now()
    allowed = {"title", "status", "assigned_to", "tags"}
    set_fields = {k: v for k, v in updates.items() if k in allowed and v is not None}

    if not set_fields:
        return get_case(db, case_id)

    set_fields["updated_at"] = now

    timeline_entries = []
    if "status" in set_fields:
        timeline_entries.append({
            "ts": now, "type": "status_change", "author": author,
            "content": f"Status changed to {set_fields['status']}",
        })
    if "assigned_to" in set_fields:
        assignee = set_fields["assigned_to"] or "unassigned"
        timeline_entries.append({
            "ts": now, "type": "note", "author": author,
            "content": f"Assigned to {assignee}",
        })

    update_op = {"$set": set_fields}
    if timeline_entries:
        update_op["$push"] = {"timeline": {"$each": timeline_entries}}

    db.cases.update_one({"_id": case_oid}, update_op)

    if "status" in set_fields:
        webhooks.deliver(db, "case.status_changed", {
            "_id": case_id,
            "title": existing_case.get("title"),
            "previous_status": existing_case.get("status"),
            "status": set_fields["status"],
        })

    return get_case(db, case_id)


def link_alert(db, case_id, alert_id, author="analyst"):
    """Links an existing alert onto a case ($addToSet -- safe to call
    twice on the same alert, won't duplicate) and logs it on the
    timeline, then recomputes the case's cached severity/entities."""
    case_oid = _oid(case_id)
    alert_oid = _oid(alert_id)

    alert = db.alerts.find_one({"_id": alert_oid})
    if not alert or not db.cases.find_one({"_id": case_oid}):
        return None

    now = _now()
    db.cases.update_one(
        {"_id": case_oid},
        {
            "$addToSet": {"alert_ids": alert_oid},
            "$push": {"timeline": {
                "ts": now, "type": "alert_linked", "author": author,
                "content": f"Linked alert: {alert.get('event_type', 'event')} on "
                           f"{alert.get('ip', '\u2014')}",
            }},
        },
    )
    _recompute_case_aggregates(db, case_oid)
    return get_case(db, case_id)


def unlink_alert(db, case_id, alert_id, author="analyst"):
    """Removes an alert from a case and logs it, then recomputes cached
    aggregates. The alert itself is untouched -- unlinking only affects
    the case's grouping, not the alert's own triage status."""
    case_oid = _oid(case_id)
    alert_oid = _oid(alert_id)

    if not db.cases.find_one({"_id": case_oid}):
        return None

    now = _now()
    db.cases.update_one(
        {"_id": case_oid},
        {
            "$pull": {"alert_ids": alert_oid},
            "$push": {"timeline": {
                "ts": now, "type": "alert_unlinked", "author": author,
                "content": f"Unlinked alert {alert_id}",
            }},
        },
    )
    _recompute_case_aggregates(db, case_oid)
    return get_case(db, case_id)


def add_note(db, case_id, content, author="analyst"):
    """Appends a free-text investigation note to the case timeline."""
    case_oid = _oid(case_id)
    if not db.cases.find_one({"_id": case_oid}):
        return None

    now = _now()
    db.cases.update_one(
        {"_id": case_oid},
        {
            "$push": {"timeline": {"ts": now, "type": "note", "author": author, "content": content}},
            "$set": {"updated_at": now},
        },
    )
    webhooks.deliver(db, "case.note_added", {"_id": case_id, "author": author, "content": content})
    return get_case(db, case_id)


def suggest_related_alerts(db, case_id, limit=10):
    """
    Auto-suggestion (not auto-linking -- see module docstring): returns
    OPEN alerts (status new/investigating) that share an IP or user_id
    with an alert already linked to this case, and aren't linked to it
    already. An analyst reviews these on the case detail view and links
    the ones that are genuinely related.
    """
    case = db.cases.find_one({"_id": _oid(case_id)})
    if not case:
        return []

    linked_ids = list(case.get("alert_ids", []))
    if not linked_ids:
        return []

    linked_alerts = list(db.alerts.find({"_id": {"$in": linked_ids}}))
    ips = list({a["ip"] for a in linked_alerts if a.get("ip")})
    users = list({a["user_id"] for a in linked_alerts if a.get("user_id")})

    if not ips and not users:
        return []

    or_clauses = []
    if ips:
        or_clauses.append({"ip": {"$in": ips}})
    if users:
        or_clauses.append({"user_id": {"$in": users}})

    query = {
        "_id": {"$nin": linked_ids},
        "status": {"$in": ["new", "investigating"]},
        "$or": or_clauses,
    }

    cursor = db.alerts.find(query).limit(limit)
    results = []
    for a in cursor:
        a["_id"] = str(a["_id"])
        results.append(a)
    return results