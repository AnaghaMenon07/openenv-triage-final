from pydantic import BaseModel


class Observation(BaseModel):
    subject: str
    body: str


class EmailTriageEnv:
    def __init__(self):
        self.current = Observation(
            subject="Welcome",
            body="Test email"
        )

    def reset(self):
        self.current = Observation(
            subject="Reset email",
            body="Hello, need help with my account"
        )
        return self.current

    def state(self):
        return self.current

    def step(self, action):
        reward = 1.0
        done = True
        info = {}

        return self.current, reward, done, info