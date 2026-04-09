import os
import requests
import json
import time

# ✅ 1. SMARTER DYNAMIC URLS
# We check for API_URL (validator), then HF_SPACE_ID (to build the URL), then localhost
API_URL = os.environ.get("API_URL")
if not API_URL:
    space_id = os.environ.get("HF_SPACE_ID") # Hugging Face automatically provides this
    if space_id:
        user, space = space_id.split("/")
        API_URL = f"https://{user}-{space.replace('_', '-')}.hf.space"
    else:
        API_URL = "http://127.0.0.1:8000"

LLM_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:4000/v1")
API_KEY = os.environ.get("API_KEY", os.getenv("OPENAI_API_KEY", "your_local_key"))

# ... (keep your simple_agent function exactly as it is) ...

def run():
    print(f"[DEBUG] Connecting to API_URL: {API_URL}", flush=True)
    print("[START] task=email_triage", flush=True)
    
    # ✅ 2. RETRY LOOP (Fixes the Connection Refused error)
    obs = None
    for attempt in range(5):
        try:
            response = requests.post(f"{API_URL}/reset", timeout=15)
            response.raise_for_status()
            obs = response.json()
            break 
        except Exception as e:
            print(f"[RETRY {attempt+1}/5] Server not ready yet... {e}", flush=True)
            time.sleep(5)

    if not obs:
        print("CRITICAL: Environment server is unreachable.", flush=True)
        return

    # Run the agent
    action = simple_agent(obs)
    
    # Step the env
    try:
        res = requests.post(f"{API_URL}/step", json=action, timeout=15).json()
        reward = res.get("reward", 0)
        print(f"[STEP] step=1 reward={reward}", flush=True)
        print(f"[END] task=email_triage score={reward} steps=1", flush=True)
    except Exception as e:
        print(f"FAILED TO STEP: {e}", flush=True)

if __name__ == "__main__":
    run()