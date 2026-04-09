import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Action(BaseModel):
    category: str
    priority: str
    response: str

class EmailTriageEnv:
    def __init__(self):
        self.current = {"subject": "Initial email", "body": "Welcome to the triage system."}

    def reset(self):
        self.current = {"subject": "Support Request", "body": "I cannot access my account."}
        return self.current

    def state(self):
        return self.current

    def step(self, action: Action):
        # Update the state to reflect the action taken
        self.current = {
            "subject": "Processed",
            "body": f"Action taken: {action.category}"
        }
        # Return observation, reward, done, info
        return self.current, 1.0, True, {}

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

def main():
    # Dynamic port for Hugging Face (7860) or local (8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()