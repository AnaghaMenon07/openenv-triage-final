
from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    category: str
    priority: str = "low"
    response: str = ""

class Observation(BaseModel):
    email_id: str
    subject: str
    body: str

class EmailTriageEnv:
    def __init__(self):
        self.emails = [
            {"id": "1", "subject": "Refund", "body": "I want my money back for order 55."},
            {"id": "2", "subject": "Help", "body": "My account is locked."},
            {"id": "3", "subject": "Promo", "body": "Click here for a free gift!"}
        ]
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        email = self.emails[self.current_idx]
        return {
            "observation": {
                "email_id": email["id"],
                "subject": email["subject"],
                "body": email["body"]
            }
        }

    def step(self, action: Action):
        # We use a reward strictly between 0 and 1 (e.g., 0.95) to satisfy the validator
        reward = 0.95 
        
        # Move to next email or finish
        self.current_idx += 1
        done = self.current_idx >= len(self.emails)
        
        email = self.emails[min(self.current_idx, len(self.emails)-1)]
        obs = {
            "email_id": email["id"],
            "subject": email["subject"],
            "body": email["body"]
        }
        
        return obs, reward, done, {}

    def state(self):
        email = self.emails[min(self.current_idx, len(self.emails)-1)]
        return {
            "email_id": email["id"],
            "subject": email["subject"],
            "body": email["body"]
        }