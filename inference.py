import os
import requests
import json
import time

# 🛑 NO FALLBACKS. We force the code to use the Validator's variables.
API_URL = os.environ.get("API_URL")
LLM_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

def simple_agent(obs):
    # This tells the validator log that we are actually trying to use their proxy
    print(f"DEBUG: Attempting LLM call to {LLM_BASE_URL}")
    
    prompt = f"Triage email. Return ONLY JSON: {{'category': '...', 'priority': '...', 'response': '...'}}. Email: {obs.get('body')}"

    # 1. THE CALL
    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "llama-3.1-8b-instant", 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        },
        timeout=30
    )
    
    # 2. THE VERIFICATION
    # If this isn't 200, the validator will now show us WHY (e.g., Wrong Model, Wrong Key)
    if response.status_code != 200:
        print(f"CRITICAL PROXY ERROR: {response.status_code} - {response.text}")
        response.raise_for_status() 

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    # Clean JSON markdown
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()

    return json.loads(content)

def run():
    print("[START] task=email_triage", flush=True)
    
    # Retry loop to wait for your HF Space to wake up
    obs = None
    for i in range(5):
        try:
            r = requests.post(f"{API_URL}/reset", timeout=10)
            r.raise_for_status()
            obs = r.json()
            break
        except:
            print(f"Waiting for HF Space... attempt {i+1}")
            time.sleep(5)

    if not obs:
        raise ConnectionError(f"Could not connect to HF Space at {API_URL}")

    # Run the agent (This is where the API call happens)
    action = simple_agent(obs)
    
    # Step the env
    res = requests.post(f"{API_URL}/step", json=action).json()
    reward = res.get("reward", 0)
    
    print(f"[STEP] step=1 reward={reward}", flush=True)
    print(f"[END] task=email_triage score={reward} steps=1", flush=True)

if __name__ == "__main__":
    run()