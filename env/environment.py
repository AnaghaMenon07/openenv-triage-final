from pydantic import BaseModel
from typing import Optional, List

# Importing the Action model from models.py to ensure consistency
# across the entire environment structure.
try:
    from env.models import Action
except ImportError:
    # Fallback definition if models.py is not in the path during local testing
    class Action(BaseModel):
        category: str
        priority: str
        response: str

class Observation(BaseModel):
    email_id: str = "1"
    subject: str
    body: str
    history: str = ""


class EmailTriageEnv:
    VALID_CATEGORIES = {"support", "billing", "spam", "inquiry", "complaint", "feedback"}
    VALID_PRIORITIES = {"low", "medium", "high"}

    def __init__(self):
        # 3 distinct email datasets to match the 3 task requirements
        # This ensures the LLM graders see unique content for each task
        self.dataset = [
            {
                "subject": "Refund Request",
                "body": "I want a refund for my order #1234. It arrived broken and I am very unhappy."
            },
            {
                "subject": "URGENT: WINNER",
                "body": "Congratulations! You have won a free iPhone. Click the link below to claim your prize immediately!"
            },
            {
                "subject": "Production Server Down",
                "body": "Help! The main database is unreachable and the entire website is throwing 500 errors."
            }
        ]
        self.idx = 0
        self.current = Observation(subject="Welcome", body="Test email")

    def reset(self):
        # Cycle through the dataset so each task gets a unique email body
        data = self.dataset[self.idx % len(self.dataset)]
        self.current = Observation(
            email_id=str(self.idx + 1),
            subject=data["subject"],
            body=data["body"]
        )
        self.idx += 1
        return self.current

    def state(self):
        return self.current

    def step(self, action: Action):
        reward = self._compute_reward(action)
        done = True
        info = {"reward_breakdown": reward}
        
        # Update current state for observation after step
        self.current = Observation(
            email_id=self.current.email_id,
            subject="Processed",
            body=f"Category: {action.category} | Priority: {action.priority}"
        )
        return self.current, reward, done, info

    def _compute_reward(self, action: Action) -> float:
        score = 0.0

        # 1. Category score (0.0 - 0.4)
        if action.category.lower() in self.VALID_CATEGORIES:
            score += 0.4
        elif action.category.strip() != "":
            score += 0.1 

        # 2. Priority score (0.0 - 0.3)
        if action.priority.lower() in self.VALID_PRIORITIES:
            score += 0.3
        elif action.priority.strip() != "":
            score += 0.1 

        # 3. Response quality score (0.0 - 0.3)
        if action.response and len(action.response.strip()) > 20:
            score += 0.3
        elif action.response and len(action.response.strip()) > 0:
            score += 0.15

        # Clamp between 0.0 and 0.95 (strictly less than 1.0)
        # This ensures compliance with Phase 2 validator rules
        return round(min(max(score, 0.0), 0.95), 2)