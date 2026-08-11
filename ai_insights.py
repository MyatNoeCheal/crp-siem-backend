"""
Generates security insights by analyzing recent SIEM events with simple,
deterministic rules -- no external API calls, no cost.

This replaces the Anthropic-API-based version. Given the survey results
(AI Insights ranked lowest priority of the 6 dashboard modules, avg 2.68
in Q17), a free rule-based summary is a reasonable, honestly-scoped choice
here rather than paying per-call for the lowest-priority tab.

The output shape matches what the dashboard already expects, so nothing
on the frontend needs to change:
    {"summary": str, "top_risks": [str, ...], "generated_at": iso str}
"""

from datetime import datetime, timezone
from collections import Counter


def generate_insights(events):
    """
    events: list of recent event dicts from the logs collection.
    Returns: {"summary": str, "top_risks": [str, ...], "generated_at": iso str}
    """
    now = datetime.now(timezone.utc).isoformat()

    if not events:
        return {
            "summary": "No events available yet to analyze. Insights will "
                       "appear once security events have been logged.",
            "top_risks": [],
            "generated_at": now,
        }

    total = len(events)
    anomalies = [e for e in events if e.get("anomaly")]
    critical = [e for e in events if e.get("risk_level") == "Critical"]
    high = [e for e in events if e.get("risk_level") == "High"]

    category_counts = Counter(e.get("category", "uncategorized") for e in events)
    event_type_counts = Counter(e.get("event_type", "unknown") for e in events)
    ip_counts = Counter(e.get("ip") for e in events if e.get("ip"))

    # --- Build the plain-language summary ---
    anomaly_rate = (len(anomalies) / total * 100) if total else 0

    summary_parts = [
        f"Analyzed the {total} most recent security events."
    ]

    if critical:
        summary_parts.append(
            f"{len(critical)} event{'s' if len(critical) != 1 else ''} "
            f"reached Critical severity and should be reviewed first."
        )
    elif high:
        summary_parts.append(
            f"No Critical events, but {len(high)} High-severity event"
            f"{'s' if len(high) != 1 else ''} were logged."
        )
    else:
        summary_parts.append("No Critical or High-severity events in this window.")

    if category_counts:
        top_category, top_count = category_counts.most_common(1)[0]
        summary_parts.append(
            f"Most activity falls under '{top_category.replace('_', ' ')}' "
            f"({top_count} of {total} events)."
        )

    summary_parts.append(
        f"Overall anomaly rate is {anomaly_rate:.1f}% "
        f"({len(anomalies)} of {total} events flagged)."
    )

    summary = " ".join(summary_parts)

    # --- Build top risks list ---
    top_risks = []

    if critical:
        types = Counter(e.get("event_type", "unknown") for e in critical)
        most_common_type, count = types.most_common(1)[0]
        top_risks.append(
            f"{len(critical)} Critical-severity event(s) logged, most commonly "
            f"'{most_common_type}' ({count} occurrence{'s' if count != 1 else ''})."
        )

    repeat_ips = [ip for ip, count in ip_counts.items() if count >= 3]
    if repeat_ips:
        worst_ip, worst_count = ip_counts.most_common(1)[0]
        top_risks.append(
            f"IP {worst_ip} generated {worst_count} events in this window — "
            f"worth checking for repeated failed logins or scripted activity."
        )

    fraud_events = [e for e in events if e.get("category") == "fraud"]
    if fraud_events:
        total_amount = sum(e.get("amount") or 0 for e in fraud_events)
        top_risks.append(
            f"{len(fraud_events)} fraud-flagged transaction(s) totaling "
            f"${total_amount:,.2f} in this window."
        )

    admin_events = [e for e in events if e.get("category") == "admin_activity"]
    high_risk_admin = [e for e in admin_events if e.get("risk_level") in ("High", "Critical")]
    if high_risk_admin:
        top_risks.append(
            f"{len(high_risk_admin)} high-risk admin action(s) detected — "
            f"review for unauthorized configuration or role changes."
        )

    if event_type_counts:
        most_common_type, count = event_type_counts.most_common(1)[0]
        if count / total > 0.4:  # one event type dominates the window
            top_risks.append(
                f"'{most_common_type}' accounts for {count}/{total} events "
                f"({count/total*100:.0f}%) — check whether this is expected "
                f"volume or a sign of a single ongoing incident."
            )

    if not top_risks:
        top_risks.append("No significant risk patterns detected in this window.")

    return {
        "summary": summary,
        "top_risks": top_risks[:5],
        "generated_at": now,
    }