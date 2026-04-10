import os
import sys
import time
import json
import asyncio
import requests
import subprocess
from typing import List, Optional

# --- Auto-install OpenAI if missing ---
try:
    import openai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional local image name for docker-based evaluation
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# The URL of your running Hugging Face space
API_URL = os.environ.get("API_URL", "https://anaghamenon-openenv-email-triage-final.hf.space")

BENCHMARK = "email_triage_env"

# Task definitions synced with openenv.yaml
TASKS = [
    {
        "name": "customer_support_triage",
        "endpoint": "/step/customer_support_triage",
        "prompt": "Classify this customer support email. Reply ONLY with JSON: {\"category\": \"<support|billing|inquiry>\"}",
        "fallback": {"category": "support"}
    },
    {
        "name": "spam_classification",
        "endpoint": "/step/spam_classification",
        "prompt": "Classify this email and assign priority. Reply ONLY with JSON: {\"category\": \"<spam|legitimate>\", \"priority\": \"<low|medium|high>\"}",
        "fallback": {"category": "legitimate", "priority": "low"}
    },
    {
        "name": "urgency_detection",
        "endpoint": "/step/urgency_detection",
        "prompt": "Detect urgency and write a professional response. Reply ONLY with JSON: {\"category\": \"<value>\", \"priority\": \"<low|high>\", \"response\": \"<text>\"}",
        "fallback": {"category": "support", "priority": "high", "response": "Acknowledged."}
    },
]

# --- Structured Logging ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run():
    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_token_provided")
    
    all_rewards = []
    total_steps = 0
    
    # Process each task level defined in openenv.yaml
    for i, task in enumerate(TASKS, start=1):
        log_start(task=task["name"], env=BENCHMARK, model=MODEL_NAME)
        
        current_reward = 0.0
        error_msg = None
        
        try:
            # 1. Reset Environment
            reset_resp = requests.post(f"{API_URL}/reset", timeout=15).json()
            obs = reset_resp.get("observation", reset_resp)

            # 2. LLM Reasoning
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": f"{task['prompt']}\n\nEmail Subject: {obs.get('subject')}\nBody: {obs.get('body')}"
                }],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            
            action_dict = json.loads(completion.choices[0].message.content)
            
            # 3. Environment Step
            step_resp = requests.post(f"{API_URL}{task['endpoint']}", json=action_dict, timeout=15).json()
            current_reward = float(step_resp.get("reward", 0.0))
            
            log_step(step=1, action=json.dumps(action_dict), reward=current_reward, done=True, error=None)
            
        except Exception as e:
            error_msg = str(e)
            log_step(step=1, action="error", reward=0.0, done=True, error=error_msg)
        
        all_rewards.append(current_reward)
        total_steps += 1

    # Final scoring
    final_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    log_end(success=final_score > 0.5, steps=total_steps, score=final_score, rewards=all_rewards)

if __name__ == "__main__":
    asyncio.run(run())