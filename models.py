from pydantic import BaseModel
from typing import Optional


class LogData(BaseModel):
    ip: str
    event_type: str
    severity: str
    timestamp: str
    user_id: Optional[str] = None
    role: Optional[str] = "user"          # "user" | "admin"
    category: Optional[str] = None        # auto-set by detector if not provided
    amount: Optional[float] = None        # for transaction/fraud events


class LogEvent(BaseModel):
    ip: str
    event_type: str
    severity: str
    timestamp: str
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = "user"          # "user" | "admin"
    category: Optional[str] = None        # threat | fraud | user_behavior | admin_activity
    amount: Optional[float] = None        # transaction amount, if applicable
    device_id: Optional[str] = None