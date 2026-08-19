from collections import defaultdict
from datetime import datetime, timedelta

# Event types that map to each dashboard tab. Extend as new event_type
# values are introduced by the e-commerce site or CERT dataset ingestion.
CATEGORY_RULES = {
    "fraud": {"payment", "transaction", "refund", "chargeback"},
    "admin_activity": {"admin_login", "config_change", "role_change", "user_delete"},
    "user_behavior": {
        "login", "logout", "profile_update", "password_change",
        # --- e-commerce shopping-flow events (added after
        # simulate_ecommerce_traffic.py surfaced these silently defaulting
        # to "threat" -- see mitre_attack.py's _MITRE_MAP, which already
        # explicitly treats these as benign, not attack techniques). These
        # feed the User Behavior tab and the LSTM sequence model
        # (lstm_inference.py) the same way login/logout do -- more
        # behavioral events per user session means a better-informed
        # sequence score, not a diluted one. ---
        "browse", "add_to_cart", "checkout",
    },
}
# Anything not matched above (e.g. failed_login, port_scan, ddos_attempt)
# defaults to "threat" — see _infer_category().

# =========================
# FALSE-POSITIVE REDUCTION TUNING
# =========================
# The IT Lecturer survey responses (Q20, n=5) independently and repeatedly
# named high false-positive rates / alert fatigue as the biggest limitation
# of current monitoring tools -- 4 of 5 lecturer free-text answers raised it
# unprompted. The two constants blocks below fix the two rules that were
# most likely causing it in the ORIGINAL rule set:
#
#   1. Failed-login counting had no time window (a plain Counter that only
#      ever went up), so an IP that failed login 5 times total -- even
#      spread across separate days -- stayed permanently flagged. Fixed
#      below with a sliding time window.
#   2. The "high-value transaction" rule used a flat $1000 cutoff. For an
#      e-commerce store selling electronics/tech accessories, a legitimate
#      laptop or monitor purchase routinely clears $1000 -- this was
#      almost certainly generating false positives on real purchases.
#      Fixed below with a threshold computed from this store's own recent
#      transaction distribution instead of an arbitrary round number.
#
# A THIRD source of false positives, found via simulate_ecommerce_traffic.py
# during e-commerce integration testing: browse/add_to_cart/checkout had no
# CATEGORY_RULES entry at all, so every normal shopping session's browsing
# activity silently defaulted to category="threat" and flooded the Threats
# tab with benign traffic. Fixed above by adding them to "user_behavior".

FAILED_LOGIN_WINDOW_MINUTES = 10
FAILED_LOGIN_THRESHOLD = 5

DEFAULT_HIGH_VALUE_THRESHOLD = 1000   # fallback when there isn't enough history yet
HIGH_VALUE_PERCENTILE = 90            # flag amounts at/above this percentile of recent transactions
HIGH_VALUE_LOOKBACK = 500             # how many recent fraud-category events to compute it from
MIN_SAMPLES_FOR_PERCENTILE = 30       # below this, a percentile is too noisy -- use the flat default instead


class AnomalyDetector:

    def __init__(self, db=None):
        # ip -> list of datetimes of recent failed_login events, pruned to
        # a sliding window on each check instead of counted forever.
        self.failed_login_events = defaultdict(list)

        # Optional Mongo db handle (see database.py's get_db()), used only
        # for the dynamic high-value-transaction threshold below. Detector
        # still works with db=None -- it just falls back to the flat
        # DEFAULT_HIGH_VALUE_THRESHOLD, same as the original behavior.
        self.db = db

    def _infer_category(self, event):
        # Respect an explicit category if the caller already set one
        if getattr(event, "category", None):
            return event.category

        # Admin role activity is routed to Admin Activity regardless of event_type
        if getattr(event, "role", "user") == "admin":
            return "admin_activity"

        for category, event_types in CATEGORY_RULES.items():
            if event.event_type in event_types:
                return category

        return "threat"

    @staticmethod
    def _parse_event_ts(event):
        """Uses the event's own timestamp (not wall-clock) so scoring is
        consistent whether events arrive live or are replayed/backfilled."""
        raw = getattr(event, "timestamp", None)
        if not raw:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.utcnow()

    def _failed_login_count_in_window(self, ip, now):
        """Counts failed_login events for this IP within the last
        FAILED_LOGIN_WINDOW_MINUTES, pruning older entries as it goes --
        replaces the old unbounded Counter that never decayed."""
        window_start = now - timedelta(minutes=FAILED_LOGIN_WINDOW_MINUTES)
        recent = [t for t in self.failed_login_events[ip] if t >= window_start]
        self.failed_login_events[ip] = recent
        return len(recent)

    def _high_value_threshold(self):
        """
        Dynamic replacement for the old flat $1000 "large transaction"
        cutoff. Computes the HIGH_VALUE_PERCENTILE-th percentile of recent
        real transaction amounts (category="fraud" events already logged),
        so the threshold reflects this store's actual spending pattern
        instead of an arbitrary round number. Falls back to the flat
        default when there isn't enough history yet (cold start / mostly
        demo data) -- MIN_SAMPLES_FOR_PERCENTILE guards against computing
        a noisy percentile off a handful of events.
        """
        if self.db is None:
            return DEFAULT_HIGH_VALUE_THRESHOLD

        try:
            amounts = [
                d["amount"] for d in self.db.logs.find(
                    {"category": "fraud", "amount": {"$ne": None}}, {"amount": 1}
                ).sort("_id", -1).limit(HIGH_VALUE_LOOKBACK)
                if d.get("amount") is not None
            ]
        except Exception:
            return DEFAULT_HIGH_VALUE_THRESHOLD

        if len(amounts) < MIN_SAMPLES_FOR_PERCENTILE:
            return DEFAULT_HIGH_VALUE_THRESHOLD

        amounts.sort()
        idx = min(int(len(amounts) * HIGH_VALUE_PERCENTILE / 100), len(amounts) - 1)
        # Never drop below half the flat default -- keeps the threshold sane
        # even if the recent distribution is unusually skewed low.
        return max(amounts[idx], DEFAULT_HIGH_VALUE_THRESHOLD * 0.5)

    def analyze(self, event):

        score = 0
        reasons = []
        now = self._parse_event_ts(event)

        # Rule 1: Multiple failed logins -- now windowed (see docstring above)
        if event.event_type == "failed_login":
            self.failed_login_events[event.ip].append(now)
            count = self._failed_login_count_in_window(event.ip, now)

            if count >= FAILED_LOGIN_THRESHOLD:
                score += 50
                reasons.append(
                    f"{count} failed logins in {FAILED_LOGIN_WINDOW_MINUTES} min"
                )

        # Rule 2: High severity event (caller-asserted)
        if event.severity.lower() == "high":
            score += 30
            reasons.append("High severity event")

        # Rule 3: Internal network activity
        if event.ip.startswith("192.168"):
            score += 10
            reasons.append("Internal network activity")

        # Rule 4: Large transaction amount -- now a dynamic, distribution-
        # aware threshold instead of a flat $1000 cutoff (see
        # _high_value_threshold docstring above)
        amount = getattr(event, "amount", None)
        if amount is not None:
            threshold = self._high_value_threshold()
            if amount >= threshold:
                score += 25
                reasons.append(
                    f"High-value transaction (${amount:,.2f}, "
                    f"threshold ${threshold:,.2f})"
                )

        category = self._infer_category(event)

        return score, reasons, category