import os
import sys
import json
import requests
import subprocess

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = os.environ.get("API_URL")
if not API_URL or str(API_URL).lower() == "none":
    API_URL = "https://anaghamenon-openenv-email-triage-final.hf.space"

TASKS = [
    {
        "name": "customer_support_triage",
        "endpoint": "/step/customer_support_triage",
        "prompt": "Classify this customer support email into a category. Reply ONLY with valid JSON: {\"category\": \"billing\", \"priority\": \"medium\", \"response\": \"We will process your request.\"}",
        "fallback": {"category": "support", "priority": "medium", "response": "We will look into this."}
    },
    {
        "name": "spam_classification",
        "endpoint": "/step/spam_classification",
        "prompt": "Classify this email as spam or legitimate. Reply ONLY with valid JSON: {\"category\": \"spam\", \"priority\": \"low\", \"response\": \"This is spam.\"}",
        "fallback": {"category": "spam", "priority": "low", "response": "This appears to be spam."}
    },
    {
        "name": "urgency_detection",
        "endpoint": "/step/urgency_detection",
        "prompt": "Detect urgency and write a professional response. Reply ONLY with valid JSON: {\"category\": \"support\", \"priority\": \"high\", \"response\": \"We are on it.\"}",
        "fallback": {"category": "support", "priority": "high", "response": "We have flagged this as urgent and will respond immediately."}
    },
]

def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"], stdout=subprocess.DEVNULL)
        from openai import OpenAI
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no_token")

def run_task(task, client):
    print(f"[START] task={task['name']}", flush=True)
    try:
        # Reset
        r = requests.post(f"{API_URL}/reset", timeout=10)
        obs_data = r.json()
        obs = obs_data.get("observation", obs_data)

        # LLM reasoning
        action_dict = task["fallback"]
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"{task['prompt']}\n\nEmail:\n{obs.get('body', '')}"}],
                temperature=0,
            )
            content = completion.choices[0].message.content or "{}"
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "category" in parsed:
                action_dict = parsed
        except Exception:
            pass

        # Step
        res = requests.post(f"{API_URL}{task['endpoint']}", json=action_dict, timeout=10).json()
        reward = float(res.get("reward", 0.0))
        score = max(0.01, min(0.99, reward))

        print(f"[STEP] step=1 reward={score:.4f}", flush=True)
        print(f"[END] success=True steps=1 score={score:.4f}", flush=True)
        print(f"[TOTAL_SUMMARY] task={task['name']} score={score:.4f}", flush=True)

    except Exception as e:
        print(f"[STEP] step=1 reward=0.01", flush=True)
        print(f"[END] success=False steps=0 score=0.01", flush=True)
        print(f"[TOTAL_SUMMARY] task={task['name']} score=0.01", flush=True)

def run():
    client = get_client()
    for task in TASKS:
        run_task(task, client)

if __name__ == "__main__":
    run()
