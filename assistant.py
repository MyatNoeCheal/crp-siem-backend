"""
JARVIS-style AI SOC Assistant -- /assistant/chat

WHY THIS FILE: the dashboard needed a conversational, voice-driven layer on
top of the existing SIEM data (entity_risk.py, cases.py, ai_insights.py,
db.alerts/db.logs) -- something an analyst can talk to, that can both
ANSWER questions ("show me critical threats") and ACT ("open the Threats
tab", "mark this alert as investigating"), the way Iron Man's JARVIS
narrates and drives the HUD rather than just chatting in a box.

ARCHITECTURE (deliberately simple, deliberately local):
  Browser (mic/speaker via Web Speech API)
    -> POST /assistant/chat  (this file, running in your existing FastAPI app)
    -> local Ollama (http://localhost:11434) running a Qwen model, using
       Ollama's OpenAI-style tool-calling
    -> tool functions below, which read/write the SAME Mongo collections
       the rest of the dashboard already uses (no new data model)

This does NOT use the OpenJarvis Python package. OpenJarvis is a
local-first *agent CLI framework* built around `jarvis init` / Ollama on
your own machine -- it isn't meant to be imported into a web backend and
driven by HTTP requests from a React app. Re-implementing just the piece
we need (an LLM tool-calling loop against our own tools) is a few dozen
lines and avoids fighting someone else's CLI-oriented framework. State
this honestly in your report if asked to justify the choice.

SETUP (local machine, once):
    1. Install Ollama: https://ollama.com
    2. ollama pull qwen2.5          # or qwen2.5:14b if you have the VRAM
    3. ollama serve                  # usually already running as a service
    4. pip install requests          # almost certainly already installed

WIRE-UP (in main.py):
    from assistant import router as assistant_router
    app.include_router(assistant_router)

ENV VARS (optional, sensible defaults):
    OLLAMA_HOST   default "http://localhost:11434"
    OLLAMA_MODEL  default "qwen2.5"

SCOPE / HONEST LIMITATIONS (worth a line in your report):
  - "Actions" JARVIS can take are limited to what the SIEM backend already
    supports: navigating the dashboard, and updating alert/case status.
    There is no real firewall integration -- "isolate this host" / "block
    this IP" are demo-narration only, exactly like the rule-based/AI
    detection itself is not wired to a real EDR. Don't imply otherwise in
    the report.
  - Tool-calling quality depends on the Qwen checkpoint. qwen2.5 (7b+)
    reliably emits tool calls via Ollama; very small quantized models may
    not.
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import requests
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from bson import ObjectId

from database import get_db
from auth import get_current_user
import entity_risk
import ai_insights

router = APIRouter(prefix="/assistant", tags=["assistant"])

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")

# Per-request timeout (seconds) for each call to Ollama's /api/chat.
# The tool-calling loop below can make several of these calls in a row
# (up to 5 rounds), and on CPU-only inference a single round with the
# full TOOLS schema attached can easily take 20-30+ seconds -- the old
# flat 60s default was tight enough to intermittently trip
# requests.exceptions.ReadTimeout on a normal, successful exchange, not
# just on genuine hangs. Configurable via env var so this can be tuned
# per machine (GPU vs CPU) without editing code.
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))

VALID_PAGES = [
    "overview", "threats", "cases", "fraud",
    "user-behavior", "admin-activity", "ai-insights", "logs",
    "reports", "settings",
]

SYSTEM_PROMPT = (
    "You are JARVIS, the AI assistant embedded in a Security Operations "
    "Center (SOC) dashboard for an e-commerce platform called LectroHub. "
    "You speak the way JARVIS does in Iron Man: concise, calm, a little "
    "formal, never chatty filler. You have real tools that read the live "
    "security database and can move the analyst around the dashboard -- "
    "use them instead of guessing. When a tool returns data, summarize it "
    "in plain spoken English (this may be read aloud by text-to-speech, "
    "so avoid bullet points, tables, or markdown -- speak in short "
    "sentences). When the analyst asks to see/open/go to a section of the "
    "dashboard, call navigate_page. Never claim to have taken a real-world "
    "action (like actually blocking an IP at a firewall) -- this system "
    "can only update records inside the SIEM itself; phrase those actions "
    "accordingly, e.g. 'I've flagged that alert as investigating' rather "
    "than 'I've blocked the attacker'."
)


# =========================
# Tool implementations -- each reads/writes the SAME collections the rest
# of the dashboard uses. Kept intentionally small and safe.
# =========================

def _now():
    return datetime.now(timezone.utc).isoformat()


def tool_navigate_page(page: str) -> dict:
    page = (page or "").strip().lower()
    if page not in VALID_PAGES:
        return {"ok": False, "error": f"'{page}' isn't a page I recognize. "
                                       f"Valid pages: {', '.join(VALID_PAGES)}"}
    return {"ok": True, "page": page}


def tool_get_dashboard_overview(db) -> dict:
    total = db.logs.count_documents({})
    critical = db.logs.count_documents({"risk_level": "Critical"})
    anomalies = db.logs.count_documents({"anomaly": True})
    by_category = {}
    for cat in ("threat", "fraud", "user_behavior", "admin_activity"):
        by_category[cat] = db.logs.count_documents({"category": cat})
    return {
        "total_events": total,
        "critical_events": critical,
        "anomalies": anomalies,
        "events_by_category": by_category,
    }


def tool_get_recent_threats(db, limit: int = 5, severity: Optional[str] = None) -> dict:
    query = {}
    if severity:
        query["risk_level"] = severity.capitalize()
    cursor = (
        db.alerts.find(query)
        .sort([("priority_score", -1), ("last_seen", -1)])
        .limit(max(1, min(limit, 20)))
    )
    results = []
    for a in cursor:
        results.append({
            "id": str(a["_id"]),
            "ip": a.get("ip"),
            "event_type": a.get("event_type"),
            "risk_level": a.get("risk_level"),
            "status": a.get("status", "new"),
            "count": a.get("count", 1),
            "last_seen": a.get("last_seen") or a.get("timestamp"),
        })
    return {"count": len(results), "threats": results}


def tool_get_fraud_summary(db, limit: int = 5) -> dict:
    cursor = db.logs.find({"category": "fraud"}).sort("_id", -1).limit(max(1, min(limit, 20)))
    results, total_amount = [], 0.0
    for e in cursor:
        amount = e.get("amount") or 0
        total_amount += amount
        results.append({
            "user_id": e.get("user_id"),
            "ip": e.get("ip"),
            "amount": amount,
            "risk_score": e.get("risk_score"),
            "risk_level": e.get("risk_level"),
            "timestamp": e.get("timestamp"),
        })
    return {"count": len(results), "total_amount": round(total_amount, 2), "transactions": results}


def tool_get_top_risk_entities(db, limit: int = 5) -> dict:
    return {"entities": entity_risk.get_top_risk_entities(db, limit=max(1, min(limit, 20)))}


def tool_get_escalating_entities(db, limit: int = 5) -> dict:
    return {"entities": entity_risk.get_escalating_entities(db, limit=max(1, min(limit, 20)))}


def tool_get_ai_insights(db) -> dict:
    events = list(db.logs.find({}).sort("_id", -1).limit(100))
    return ai_insights.generate_insights(events)


def tool_update_alert_status(db, alert_id: str, status: str) -> dict:
    valid = {"new", "investigating", "resolved", "false_positive"}
    if status not in valid:
        return {"ok": False, "error": f"status must be one of {sorted(valid)}"}
    try:
        oid = ObjectId(alert_id)
    except Exception:
        return {"ok": False, "error": f"'{alert_id}' isn't a valid alert id"}
    result = db.alerts.update_one({"_id": oid}, {"$set": {"status": status, "updated_at": _now()}})
    if result.matched_count == 0:
        return {"ok": False, "error": f"No alert found with id {alert_id}"}
    return {"ok": True, "alert_id": alert_id, "status": status}


# =========================
# Tool schema (Ollama / OpenAI-style function-calling format)
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_page",
            "description": "Switch the dashboard to a different page/tab for the analyst. Use whenever the analyst asks to see, open, go to, or switch to a section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "enum": VALID_PAGES,
                        "description": "The dashboard page to navigate to.",
                    }
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dashboard_overview",
            "description": "Get high-level counts: total events, critical events, AI-flagged anomalies, and event counts per category.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_threats",
            "description": "Get the most recent/highest-priority security threat alerts, optionally filtered by severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many to return, default 5, max 20"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fraud_summary",
            "description": "Get recent flagged fraudulent transactions and their total dollar amount.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "default 5, max 20"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_entities",
            "description": "Get the IPs/users with the highest current UEBA risk score (persistent, time-decaying risk).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "default 5, max 20"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_escalating_entities",
            "description": "Get entities (IPs/users) whose risk is climbing fast right now but hasn't triggered a full alert yet -- an early warning list.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "default 5, max 20"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ai_insights",
            "description": "Get a plain-language rule-based summary and top risks across the most recent events.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_alert_status",
            "description": "Update the triage status of a specific alert in the SIEM (e.g. mark as investigating or resolved). This does NOT take any real-world network action like actually blocking traffic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alert_id": {"type": "string", "description": "The alert's _id, from a prior get_recent_threats call"},
                    "status": {"type": "string", "enum": ["new", "investigating", "resolved", "false_positive"]},
                },
                "required": ["alert_id", "status"],
            },
        },
    },
]


def _dispatch_tool(db, name: str, args: dict) -> dict:
    try:
        if name == "navigate_page":
            return tool_navigate_page(args.get("page", ""))
        if name == "get_dashboard_overview":
            return tool_get_dashboard_overview(db)
        if name == "get_recent_threats":
            return tool_get_recent_threats(db, args.get("limit", 5), args.get("severity"))
        if name == "get_fraud_summary":
            return tool_get_fraud_summary(db, args.get("limit", 5))
        if name == "get_top_risk_entities":
            return tool_get_top_risk_entities(db, args.get("limit", 5))
        if name == "get_escalating_entities":
            return tool_get_escalating_entities(db, args.get("limit", 5))
        if name == "get_ai_insights":
            return tool_get_ai_insights(db)
        if name == "update_alert_status":
            return tool_update_alert_status(db, args.get("alert_id", ""), args.get("status", ""))
        return {"ok": False, "error": f"Unknown tool '{name}'"}
    except Exception as e:
        return {"ok": False, "error": f"Tool '{name}' failed: {e}"}


# =========================
# Ollama tool-calling loop
# =========================

def _ollama_chat(messages: List[dict]) -> dict:
    resp = requests.post(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    current_page: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    action: Optional[Dict[str, Any]] = None
    tool_calls: List[str] = []


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_current_user)])
def assistant_chat(req: ChatRequest):
    db = get_db()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.current_page:
        messages.append({
            "role": "system",
            "content": f"The analyst is currently looking at the '{req.current_page}' page.",
        })
    for m in req.history[-12:]:  # keep context small; this is a live voice loop, not a research session
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    action = None
    tool_calls_made = []

    # Tool-calling loop: give the model up to a few rounds to call tools
    # and read the results before producing its final spoken answer.
    for _ in range(5):
        data = _ollama_chat(messages)
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return ChatResponse(
                reply=msg.get("content", "").strip() or "I don't have anything to add.",
                action=action,
                tool_calls=tool_calls_made,
            )

        messages.append(msg)
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

            result = _dispatch_tool(db, name, args)
            tool_calls_made.append(name)

            if name == "navigate_page" and result.get("ok"):
                action = {"type": "navigate", "target": result["page"]}

            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str),
            })

    # Safety valve: if the model won't stop calling tools, force a final answer.
    messages.append({
        "role": "system",
        "content": "Give your final spoken answer now, in plain sentences, no more tool calls.",
    })
    data = _ollama_chat(messages)
    return ChatResponse(
        reply=data.get("message", {}).get("content", "").strip() or "Done.",
        action=action,
        tool_calls=tool_calls_made,
    )


@router.get("/health")
def assistant_health():
    """Quick check the browser can call before enabling voice mode, so a
    cold/offline Ollama fails with a clear message instead of a silent
    30s timeout mid-conversation."""
    try:
        r = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model_ready = any(OLLAMA_MODEL in m for m in models)
        return {
            "ollama_reachable": True,
            "model": OLLAMA_MODEL,
            "model_pulled": model_ready,
            "available_models": models,
        }
    except Exception as e:
        return {"ollama_reachable": False, "error": str(e), "model": OLLAMA_MODEL}