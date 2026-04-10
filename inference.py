import os
import sys
import json
import asyncio
import requests
import subprocess
from typing import List, Optional

# 1. Environment Variable Discovery
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_URL = os.environ.get("API_URL")
if not API_URL or str(API_URL).lower() == "none":
    API_URL = "https://anaghamenon-openenv-email-triage-final.hf.space"

TASK_NAME = "email_triage"
BENCHMARK = "openenv_scaler"

# SYNCED: Endpoints and prompts matched to your dynamic environment
TASKS = [
    {
        "name": "customer_support_triage",
        "endpoint": "/step/customer_support_triage",
        "prompt": "Classify this email. Return valid JSON with 'category' (e.g. billing, support, technical).",
        "fallback": {"category": "support"}
    },
    {
        "name": "spam_classification",
        "endpoint": "/step/spam_classification",
        "prompt": "Identify if this is spam. Return valid JSON with 'category' and 'priority'.",
        "fallback": {"category": "support", "priority": "low"}
    },
    {
        "name": "urgency_detection",
        "endpoint": "/step/urgency_detection",
        "prompt": "Write a professional response and detect priority. Return JSON with 'category', 'priority', and 'response'.",
        "fallback": {"category": "support", "priority": "high", "response": "Processing your request."}
    },
]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run():
    # Shielded import for OpenAI
    client = None
    import_error_msg = None

    try:
        try:
            from openai import OpenAI
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"], stdout=subprocess.DEVNULL)
            from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_token")
    except Exception as e:
        import_error_msg = f"Init failed: {str(e)[:30]}"

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    all_rewards = []
    steps_taken = 0

    for i, task in enumerate(TASKS, start=1):
        obs = None
        current_step_error = import_error_msg
        
        # 1. Reset
        try:
            r = requests.post(f"{API_URL}/reset", timeout=10)
            if r.status_code == 200:
                obs_data = r.json()
                obs = obs_data.get("observation", obs_data)
        except Exception as e:
            current_step_error = f"Reset failed: {str(e)[:30]}"

        if not obs:
            log_step(step=i, action="reset_failed", reward=0.0, done=True, error=current_step_error)
            all_rewards.append(0.0)
            continue

        # 2. Reasoning
        action_dict = task["fallback"]
        if client:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": f"{task['prompt']}\n\nEmail Body: {obs.get('body', '')}"}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                content = completion.choices[0].message.content
                action_dict = json.loads(content)
            except Exception as e:
                current_step_error = f"LLM failed: {str(e)[:30]}"

        # 3. Step (Dynamic Reward)
        reward = 0.0
        try:
            res = requests.post(f"{API_URL}{task['endpoint']}", json=action_dict, timeout=10).json()
            # We take the actual reward from the environment, NO HARDCODED FALLBACK
            reward = float(res.get("reward", 0.0))
        except Exception as e:
            current_step_error = f"Step failed: {str(e)[:30]}"

        all_rewards.append(reward)
        steps_taken = i
        log_step(step=i, action=json.dumps(action_dict), reward=reward, done=True, error=current_step_error)

    final_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    log_end(success=(final_score > 0.1), steps=steps_taken, score=final_score, rewards=all_rewards)

if __name__ == "__main__":
    asyncio.run(run())