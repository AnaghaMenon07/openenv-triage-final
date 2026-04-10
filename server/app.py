import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    return {"status": "online", "message": "Ready"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage-env",
        "description": "Context-aware email triage environment",
        "version": "0.1.0",
        "tasks": ["customer_support_triage", "spam_classification", "urgency_detection"]
    }

@app.get("/schema")
def schema():
    return {
        "action": {"type": "object", "properties": {"category": {"type": "string"}, "priority": {"type": "string"}, "response": {"type": "string"}}},
        "observation": {"type": "object", "properties": {"email_id": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}},
        "state": {"type": "object", "properties": {"email_id": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}},
    }

@app.post("/reset")
def reset_endpoint():
    return env.reset()

@app.get("/state")
def state_endpoint():
    return env.state()

@app.post("/step/customer_support_triage")
@app.post("/step/spam_classification")
@app.post("/step/urgency_detection")
@app.post("/step")
def step_endpoint(req: ActionRequest):
    obs, reward, done, info = env.step(Action(category=req.category, priority=req.priority, response=req.response))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.post("/grader")
async def grader_endpoint(request: Request):
    body = await request.json()
    task_name = body.get("task_name", "customer_support_triage")
    action = body.get("action", {})
    obs = env.state()
    a = Action(category=action.get("category", ""), priority=action.get("priority", "low"), response=action.get("response", ""))
    _, reward, done, info = env.step(a)
    if task_name == "spam_classification":
        from grade.spam_classification import grade
    elif task_name == "urgency_detection":
        from grade.urgency_detection import grade
    else:
        from grade.customer_support_triage import grade
    success, score = grade(obs, action, reward, done, info)
    return {"success": success, "score": score, "reward": reward}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
