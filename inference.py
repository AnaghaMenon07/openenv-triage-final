import os
import requests
import json
import time

# 1. URL Discovery
API_URL = os.environ.get("API_URL")
if not API_URL or API_URL == "None":
    API_URL = "https://anaghamenon-openenv-email-triage-final.hf.space"

# 2. Universal Proxy Config (Tries both common names)
API_BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")

# 3. Validation - If they give us nothing, we HAVE to fallback to a working default
if not API_BASE_URL:
    API_BASE_URL = "https://api.groq.com/openai/v1"

def simple_agent(obs):
    prompt = f"Triage email. Return ONLY JSON: {{'category': '...', 'priority': '...', 'response': '...'}}. Email: {obs.get('body')}"

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()

        return json.loads(content)
    except Exception as e:
        # Crash loudly so the validator sees the error in the logs
        print(f"DEBUG: Inference failed: {e}", flush=True)
        raise e

def run():
    # ✅ 2. STRICT STRUCTURED LOGGING (START/STEP/END)
    print("[START] task=email_triage", flush=True)
    
    # Connect to HF Space
    obs = None
    for i in range(5):
        try:
            r = requests.post(f"{API_URL}/reset", timeout=10)
            r.raise_for_status()
            obs = r.json()
            break
        except Exception as e:
            time.sleep(5)

    if not obs:
        raise ConnectionError(f"Could not connect to HF Space at {API_URL}")

    # Agent call
    action = simple_agent(obs)
    
    # Step the env
    res = requests.post(f"{API_URL}/step", json=action).json()
    reward = res.get("reward", 0)
    
    # ✅ 3. EXACT LOG FORMATTING
    print(f"[STEP] step=1 reward={reward}", flush=True)
    print(f"[END] task=email_triage score={reward} steps=1", flush=True)

if __name__ == "__main__":
    run()