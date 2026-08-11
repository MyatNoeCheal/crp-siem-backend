from collections import Counter

# Event types that map to each dashboard tab. Extend as new event_type
# values are introduced by the e-commerce site or CERT dataset ingestion.
CATEGORY_RULES = {
    "fraud": {"payment", "transaction", "refund", "chargeback"},
    "admin_activity": {"admin_login", "config_change", "role_change", "user_delete"},
    "user_behavior": {"login", "logout", "profile_update", "password_change"},
}
# Anything not matched above (e.g. failed_login, port_scan, ddos_attempt)
# defaults to "threat" — see _infer_category().


class AnomalyDetector:

    def __init__(self):
        self.failed_login_counter = Counter()

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

    def analyze(self, event):

        score = 0
        reasons = []

        # Rule 1: Multiple failed logins
        if event.event_type == "failed_login":
            self.failed_login_counter[event.ip] += 1

            count = self.failed_login_counter[event.ip]

            if count >= 5:
                score += 50
                reasons.append(f"{count} failed logins detected")

        # Rule 2: High severity event
        if event.severity.lower() == "high":
            score += 30
            reasons.append("High severity event")

        # Rule 3: Internal network activity
        if event.ip.startswith("192.168"):
            score += 10
            reasons.append("Internal network activity")

        # Rule 4: Large transaction amount (fraud signal)
        amount = getattr(event, "amount", None)
        if amount is not None and amount >= 1000:
            score += 25
            reasons.append(f"High-value transaction (${amount:,.2f})")

        category = self._infer_category(event)

        return score, reasons, category