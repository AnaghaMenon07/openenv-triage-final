from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    category: str
    priority: str = "low"
    response: str = ""

class EmailTriageEnv:
    def __init__(self):
        # We add 'target' labels to check against the agent's action
        self.emails = [
            {"id": "1", "subject": "Refund", "body": "I want my money back for order 55.", "target": "billing"},
            {"id": "2", "subject": "Help", "body": "My account is locked.", "target": "support"},
            {"id": "3", "subject": "Promo", "body": "Click here for a free gift!", "target": "spam"}
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
        current_email = self.emails[min(self.current_idx, len(self.emails)-1)]
        
        # --- DYNAMIC REWARD CALCULATION ---
        # 1. Base score for providing any category
        base_score = 0.4 if action.category else 0.1
        
        # 2. Accuracy bonus: check if agent's category matches our target
        accuracy_bonus = 0.4 if action.category.lower() == current_email["target"] else 0.0
        
        # 3. Quality bonus: check if they provided a response or priority
        quality_bonus = 0.15 if (action.priority != "low" or len(action.response) > 5) else 0.05
        
        # Final result is dynamic and stays strictly between 0 and 1
        reward = float(round(base_score + accuracy_bonus + quality_bonus, 2))
        reward = max(0.1, min(0.9, reward)) # Ensure it never hits 0.0 or 1.0

        self.current_idx += 1
        done = self.current_idx >= len(self.emails)
        
        next_email = self.emails[min(self.current_idx, len(self.emails)-1)]
        obs = {
            "email_id": next_email["id"],
            "subject": next_email["subject"],
            "body": next_email["body"]
        }
        
        return obs, reward, done, {"target_category": current_email["target"]}

    def state(self):
        email = self.emails[min(self.current_idx, len(self.emails)-1)]
        return {
            "email_id": email["id"],
            "subject": email["subject"],
            "body": email["body"]
        }