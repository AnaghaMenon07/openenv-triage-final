from pydantic import BaseModel
from typing import Optional


class Observation(BaseModel):
    email_id: str = "1"
    subject: str
    body: str
    history: str = ""


class Action(BaseModel):
    category: str
    priority: str = ""
    response: str = ""


class EmailTriageEnv:
    VALID_CATEGORIES = {"support", "billing", "spam", "inquiry", "complaint", "feedback"}
    VALID_PRIORITIES = {"low", "medium", "high"}

    def __init__(self):
        self.current = Observation(subject="Welcome", body="Test email")

    def reset(self):
        self.current = Observation(
            subject="Support Request",
            body="I cannot access my account. Please help."
        )
        return self.current

    def state(self):
        return self.current

    def step(self, action: Action):
        reward = self._compute_reward(action)
        done = True
        info = {"reward_breakdown": reward}
        self.current = Observation(
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
            score += 0.1  # partial credit for attempting

        # 2. Priority score (0.0 - 0.3)
        if action.priority.lower() in self.VALID_PRIORITIES:
            score += 0.3
        elif action.priority.strip() != "":
            score += 0.1  # partial credit

        # 3. Response quality score (0.0 - 0.3)
        if action.response and len(action.response.strip()) > 20:
            score += 0.3
        elif action.response and len(action.response.strip()) > 0:
            score += 0.15  # partial credit for short response

        # Clamp between 0.0 and 1.0
        return round(min(max(score, 0.0), 1.0), 2)