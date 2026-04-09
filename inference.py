import os
import requests
import json
import time

# ✅ 1. SETUP DYNAMIC URLS
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
LLM_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:4000/v1")
API_KEY = os.environ.get("API_KEY", os.getenv("OPENAI_API_KEY", "your_local_key"))

def simple_agent(obs):
    prompt = f"Triage email. Return ONLY JSON: {{'category': '...', 'priority': '...', 'response': '...'}}. Email: {obs.get('body')}"

    # ✅ 2. THE API CALL (Crucial for Phase 2)
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}" # Uses injected key
            },
            json={
                "model": "llama-3.1-8b-instant", # ✅ No 'groq/' prefix
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=30
        )
        
        # If this fails, the proxy won't log a call
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # Clean JSON markdown
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()

        return json.loads(content)

    except Exception as e:
        print(f"PROXY CALL FAILED: {e}")
        # Return a valid dict so the loop continues, but the proxy log might be empty
        return {"category": "support", "priority": "low", "response": "error"}

def run():
    print("[START] task=email_triage", flush=True)
    # Reset env
    obs = requests.post(f"{API_URL}/reset").json()
    
    # Run the agent
    action = simple_agent(obs)
    
    # Step the env
    res = requests.post(f"{API_URL}/step", json=action).json()
    
    reward = res.get("reward", 0)
    print(f"[STEP] step=1 reward={reward}", flush=True)
    print(f"[END] task=email_triage score={reward} steps=1", flush=True)

if __name__ == "__main__":
    run()