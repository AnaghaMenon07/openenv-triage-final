import os
import sys
import json
import asyncio
import requests
import subprocess
from typing import List, Optional

# --- Auto-install OpenAI if missing ---
try:
    import openai
except ImportError:
    print("Installing openai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])

from openai import OpenAI

# --- Configuration ---
# These are provided by the environment during official evaluation
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Added for Docker-based local evaluation
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# This must be the URL of your RUNNING Hugging Face space
# The validator will override this, but for local testing, ensure it matches your space URL
API_URL = os.environ.get("API_URL", "https://anaghamenon-openenv-email-triage-final.hf.space")

BENCHMARK = "email-triage-env"

# Task definitions synced with openenv.yaml
TASKS = [
    {
        "name": "customer_support_triage",
        "endpoint": "/step/customer_support_triage",
        "prompt": "Classify this customer support email. Reply ONLY with JSON: {\"category\": \"support|billing|inquiry\"}"
    },
    {
        "name": "spam_classification",
        "endpoint": "/step/spam_classification",
        "prompt": "Classify this email. Reply ONLY with JSON: {\"category\": \"spam|legitimate\", \"priority\": \"low|medium|high\"}"
    },
    {
        "name": "urgency_detection",
        "endpoint": "/step/urgency_detection",
        "prompt": "Detect urgency and write a professional response. Reply ONLY with JSON: {\"category\": \"string\", \"priority\": \"low|high\", \"response\": \"string\"}"
    },
]

# --- Mandatory Logging Functions ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_evaluation():
    # Initialize OpenAI client
    # The 'no_token_provided' fallback is what triggers your 401 locally if HF_TOKEN isn't set
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_token_provided")
    
    all_rewards = []
    total_steps_executed = 0
    
    for task in TASKS:
        log_start(task=task["name"], env=BENCHMARK, model=MODEL_NAME)
        
        current_reward = 0.0
        try:
            # 1. Reset the environment for the new task
            reset_resp = requests.post(f"{API_URL}/reset", timeout=15)
            reset_data = reset_resp.json()
            obs = reset_data.get("observation", reset_data)

            # 2. Get LLM reasoning for the action
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user", 
                    "content": f"{task['prompt']}\n\nSubject: {obs.get('subject')}\nBody: {obs.get('body')}"
                }],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            
            # Parse the LLM's suggested action
            action_content = completion.choices[0].message.content
            action_dict = json.loads(action_content)
            
            # 3. Execute the action in your environment
            step_resp = requests.post(f"{API_URL}{task['endpoint']}", json=action_dict, timeout=15)
            step_data = step_resp.json()
            
            current_reward = float(step_data.get("reward", 0.0))
            
            log_step(
                step=1, 
                action=json.dumps(action_dict), 
                reward=current_reward, 
                done=True, 
                error=None
            )
            
        except Exception as e:
            # Log errors without crashing the entire run
            log_step(step=1, action="error", reward=0.0, done=True, error=str(e))
        
        all_rewards.append(current_reward)
        total_steps_executed += 1

    # Calculate final results
    final_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    log_end(
        success=final_score >= 0.5, 
        steps=total_steps_executed, 
        score=final_score, 
        rewards=all_rewards
    )

if __name__ == "__main__":
    asyncio.run(run_evaluation())