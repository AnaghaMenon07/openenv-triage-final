import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv, Action

app = FastAPI(version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Action Models matching action_space in openenv.yaml ---
class EasyAction(BaseModel):
    category: str

class MediumAction(BaseModel):
    category: str
    priority: str

# --- Initialize env ---
env = EmailTriageEnv()

# --- Required OpenEnv Discovery Endpoints ---

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage-env",
        "description": "Context-aware email triage environment with classification, prioritization, and response generation",
        "version": "0.1.0",
        "tasks": ["customer_support_triage", "spam_classification", "urgency_detection"]
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "priority": {"type": "string"},
                "response": {"type": "string"}
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "history": {"type": "string"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "history": {"type": "string"}
            }
        }
    }

@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "email-triage-env",
            "description": "Email triage environment MCP endpoint"
        }
    }

@app.get("/")
def root():
    return {"message": "Email triage env running"}

@app.post("/reset")
def reset_endpoint():
    return env.reset()

@app.get("/state")
def state_endpoint():
    return env.state()

# --- Task Specific Step Endpoints ---

@app.post("/step/customer_support_triage")
def step_customer_support(action: EasyAction):
    full_action = Action(category=action.category, priority="low", response="")
    next_obs, reward, done, info = env.step(full_action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

@app.post("/step/spam_classification")
def step_spam(action: MediumAction):
    full_action = Action(category=action.category, priority=action.priority, response="")
    next_obs, reward, done, info = env.step(full_action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

@app.post("/step/urgency_detection")
def step_urgency(action: Action):
    next_obs, reward, done, info = env.step(action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

@app.post("/step")
def step_endpoint(action: Action):
    next_obs, reward, done, info = env.step(action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()