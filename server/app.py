import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal imports
from env.environment import EmailTriageEnv, Action

app = FastAPI(title="Email Triage OpenEnv", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActionRequest(BaseModel):
    category: str
    priority: str = "low"
    response: str = ""

env = EmailTriageEnv()

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Email Triage Environment is running!",
        "endpoints": ["/health", "/metadata", "/schema", "/reset", "/step"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage-env",
        "description": "Context-aware email triage environment",
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
                "body": {"type": "string"}
            }
        }
    }

@app.post("/reset")
def reset_endpoint():
    return env.reset()

@app.post("/step/customer_support_triage")
@app.post("/step/spam_classification")
@app.post("/step/urgency_detection")
@app.post("/step")
async def step_endpoint(req: ActionRequest):
    full_action = Action(
        category=req.category,
        priority=req.priority,
        response=req.response
    )
    next_obs, reward, done, info = env.step(full_action)
    return {"observation": next_obs, "reward": reward, "done": done, "info": info}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)