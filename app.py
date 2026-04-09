from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# ✅ define Action here (no env.models)
class Action(BaseModel):
    category: str
    priority: str
    response: str


# ✅ simple environment (no server.app import)
class EmailTriageEnv:
    def __init__(self):
        self.current = {"subject": "test", "body": "hello"}

    def reset(self):
        self.current = {"subject": "Reset email", "body": "Need help"}
        return self.current

    def state(self):
        return self.current

    def step(self, action):
        return self.current, 1.0, True, {}


# ✅ create instance AFTER class
env = EmailTriageEnv()


@app.get("/")
def root():
    return {"message": "Email triage env running"}


@app.post("/reset")
def reset():
    return env.reset()


@app.get("/state")
def state():
    return env.state()


@app.post("/step")
def step(action: Action):
    next_obs, reward, done, info = env.step(action)

    return {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "info": info
    }