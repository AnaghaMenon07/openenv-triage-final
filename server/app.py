import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import EmailTriageEnv, Action

app = FastAPI()
env = EmailTriageEnv()

class ActionRequest(BaseModel):
    category: str
    priority: str = "low"
    response: str = ""

@app.get("/health")
def health(): return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage-env",
        "tasks": ["customer_support_triage", "spam_classification", "urgency_detection"]
    }

@app.post("/reset")
def reset(): return env.reset()

@app.post("/step/customer_support_triage")
@app.post("/step/spam_classification")
@app.post("/step/urgency_detection")
@app.post("/step")
def step(req: ActionRequest):
    obs, reward, done, info = env.step(Action(category=req.category, priority=req.priority, response=req.response))
    return {"observation": obs, "reward": reward, "done": done, "info": info}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()