import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing from your project structure
from env.environment import EmailTriageEnv, Action

app = FastAPI(version="0.1.0")

# Enable CORS for Hugging Face space compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---
class ActionRequest(BaseModel):
    category: str
    priority: str = "low"
    response: str = ""

# --- Initialize environment ---
env = EmailTriageEnv()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    # MANDATORY: These names must match your openenv.yaml tasks EXACTLY
    return {
        "name": "email-triage-env",
        "description": "Context-aware email triage environment",
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
                "body": {"type": "string"}
            }
        }
    }

@app.post("/reset")
def reset_endpoint():
    return env.reset()

# --- Task-Specific Step Endpoints ---
@app.post("/step/customer_support_triage")
@app.post("/step/spam_classification")
@app.post("/step/urgency_detection")
@app.post("/step")
async def step_endpoint(req: ActionRequest):
    # Convert Pydantic request to the 'Action' object your environment expects
    full_action = Action(
        category=req.category,
        priority=req.priority,
        response=req.response
    )
    
    # Run the step in your environment
    next_obs, reward, done, info = env.step(full_action)
    
    return {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    # Hugging Face Spaces always use port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()