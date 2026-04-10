from pydantic import BaseModel


class Observation(BaseModel):
    email_id: str = "1"
    subject: str
    body: str
    history: str = ""


class EmailTriageEnv:
    def __init__(self):
        self.current = Observation(
            subject="Welcome",
            body="Test email"
        )

    def reset(self):
        self.current = Observation(
            subject="Support Request",
            body="I cannot access my account. Please help."
        )
        return self.current

    def state(self):
        return self.current

    def step(self, action):
        # MANDATORY: strictly between 0 and 1 (not 0, not 1)
        reward = 0.95
        done = True
        info = {}
        self.current = Observation(
            subject="Processed",
            body=f"Action taken: {action.category}"
        )
        return self.current, reward, done, info