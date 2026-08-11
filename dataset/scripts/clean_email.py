import pandas as pd
import os


def get_logs():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "cert", "email.csv")

    print("Loading file:", file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    df = df[
        ["date", "user", "pc", "from", "to", "size", "attachments"]
    ]

    df["date"] = pd.to_datetime(df["date"])
    df.fillna("", inplace=True)

    df["event_type"] = "email"

    def severity(row):
        try:
            return "medium" if int(row["attachments"]) > 0 else "low"
        except:
            return "low"

    df["severity"] = df.apply(severity, axis=1)

    def transform(row):
        return {
            "timestamp": row["date"].isoformat(),
            "user_id": row["user"],
            "device_id": row["pc"],
            "event_type": row["event_type"],
            "email_from": row["from"],
            "email_to": row["to"],
            "email_size": int(row["size"]) if row["size"] != "" else 0,
            "attachments": int(row["attachments"]) if row["attachments"] != "" else 0,
            "severity": row["severity"],
            "source": "CERT_EMAIL"
        }

    return df.apply(transform, axis=1).tolist()