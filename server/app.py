import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import from env/ folder — single source of truth
from env.environment import EmailTriageEnv
from env.models import Action
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Additional models for easy/medium tasks ---
class EasyAction(BaseModel):
    category: str

class MediumAction(BaseModel):
    category: str
    priority: str

# Initialize env
env = EmailTriageEnv()

@app.get("/")
def root():
    return {"message": "Email triage env running"}

@app.post("/reset")
def reset_endpoint():
    return env.reset()

@app.get("/state")
def state_endpoint():
    return env.state()

# Easy task — category only
@app.post("/step/easy")
def step_easy(action: EasyAction):
    full_action = Action(category=action.category, priority="low", response="N/A")
    next_obs, reward, done, info = env.step(full_action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

# Medium task — category + priority
@app.post("/step/medium")
def step_medium(action: MediumAction):
    full_action = Action(category=action.category, priority=action.priority, response="N/A")
    next_obs, reward, done, info = env.step(full_action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

# Hard task — full action
@app.post("/step/hard")
def step_hard(action: Action):
    next_obs, reward, done, info = env.step(action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

# Generic /step fallback
@app.post("/step")
def step_endpoint(action: Action):
    next_obs, reward, done, info = env.step(action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()