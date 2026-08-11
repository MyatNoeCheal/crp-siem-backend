def normalize_log(log):
    return {
        "ip": log.ip,
        "event_type": log.event_type.lower(),
        "severity": log.severity.lower(),
        "timestamp": log.timestamp
    }

def get_risk_level(score: int):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    elif score < 85:
        return "High"
    else:
        return "Critical"