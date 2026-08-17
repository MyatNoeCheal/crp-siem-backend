"""
Maps SIEM event types to MITRE ATT&CK (Enterprise) tactics and techniques,
so alerts on the Threats/Admin Activity tabs show an analyst-recognizable
TTP label instead of just a raw event_type string.

WHY THIS MATTERS FOR THE REPORT: real SOC tooling (Splunk ES, QRadar,
Wazuh, Microsoft Sentinel) all map detections to ATT&CK -- it's the
industry-standard vocabulary for describing *what stage* of an attack
something represents, not just its severity. Adding this costs no extra
model training, but demonstrates SOC-tooling maturity for the Critical
Evaluation section, and gives you a legitimate hook to cite MITRE ATT&CK
directly in your Literature Review / methodology.

HONEST SCOPE NOTE (state this in the report, same pattern as the other
honesty notes throughout this codebase): MITRE ATT&CK's Enterprise
matrix models cyber intrusion behavior (initial access, credential
theft, lateral movement, etc.) -- it does NOT model payment/card fraud,
which is a financial-crime pattern, not an intrusion TTP. Fraud-category
events (payment, transaction, refund, chargeback) are therefore
deliberately NOT mapped to ATT&CK here; they continue to be handled by
the Fraud Autoencoder's own anomaly scoring instead. Force-mapping fraud
onto ATT&CK anyway would be a cosmetic stretch of the framework, not a
genuine fit -- worth stating plainly rather than implying full coverage.

Source: MITRE ATT&CK Enterprise Matrix (https://attack.mitre.org)
"""

# event_type (lowercase) -> ATT&CK mapping. Event types not present here
# (fraud events, and benign user actions like "browse", "login",
# "add_to_cart") intentionally return None -- not every event is an
# attack technique, and forcing a mapping would misrepresent the data.
_MITRE_MAP = {
    # --- Threats: reconnaissance, initial access, credential access ---
    "port_scan": {
        "tactic": "Reconnaissance", "tactic_id": "TA0043",
        "technique": "Active Scanning: Vulnerability Scanning", "technique_id": "T1595.002",
    },
    "failed_login": {
        "tactic": "Credential Access", "tactic_id": "TA0006",
        "technique": "Brute Force: Password Guessing", "technique_id": "T1110.001",
    },
    "brute_force": {
        "tactic": "Credential Access", "tactic_id": "TA0006",
        "technique": "Brute Force", "technique_id": "T1110",
    },
    "network_intrusion": {
        "tactic": "Initial Access", "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application", "technique_id": "T1190",
    },
    "sql_injection": {
        "tactic": "Initial Access", "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application", "technique_id": "T1190",
    },
    "xss_attempt": {
        "tactic": "Initial Access", "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application", "technique_id": "T1190",
    },

    # --- Admin activity: privilege escalation, defense evasion, impact ---
    "admin_login": {
        "tactic": "Initial Access", "tactic_id": "TA0001",
        "technique": "Valid Accounts", "technique_id": "T1078",
    },
    "role_change": {
        "tactic": "Privilege Escalation", "tactic_id": "TA0004",
        "technique": "Account Manipulation", "technique_id": "T1098",
    },
    "config_change": {
        "tactic": "Defense Evasion", "tactic_id": "TA0005",
        "technique": "Impair Defenses", "technique_id": "T1562",
    },
    "password_change": {
        "tactic": "Persistence", "tactic_id": "TA0003",
        "technique": "Account Manipulation", "technique_id": "T1098",
    },
    "user_delete": {
        "tactic": "Impact", "tactic_id": "TA0040",
        "technique": "Account Access Removal", "technique_id": "T1531",
    },

    # --- Benign / not attack techniques -- explicit None, not just absent ---
    "profile_update": None,
    "login": None,
    "logout": None,
    "browse": None,
    "add_to_cart": None,
    "checkout": None,
}


def get_mitre_mapping(event_type, category=None):
    """
    Returns {"tactic", "tactic_id", "technique", "technique_id", "url"}
    for a given event_type, or None if this event type isn't modeled as
    an ATT&CK technique (fraud events, benign user actions, or anything
    not in the map).
    """
    if category == "fraud":
        return None  # see module docstring -- deliberately not mapped

    mapping = _MITRE_MAP.get((event_type or "").lower())
    if not mapping:
        return None

    result = dict(mapping)
    result["url"] = (
        f"https://attack.mitre.org/techniques/"
        f"{mapping['technique_id'].replace('.', '/')}/"
    )
    return result