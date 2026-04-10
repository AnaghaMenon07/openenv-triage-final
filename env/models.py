from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    """
    Schema for actions sent to the environment.
    'category' is the primary field used for triage (e.g., support, billing, spam).
    """
    category: str
    priority: str = "low"
    response: str = ""

class Observation(BaseModel):
    """
    Schema for observations returned by the environment.
    Represents the current state of the email being triaged.
    """
    email_id: str
    subject: str
    body: str
    history: Optional[str] = ""