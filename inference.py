import os
import requests
import json

# ✅ CRITICAL: Using environment variables provided by the validator
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
LLM_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:4000/v1")
API_KEY = os.environ.get("API_KEY", os.getenv("OPENAI_API_KEY", "your_local_key"))

def simple_agent(obs):
    prompt = f"Triage this email. Return ONLY JSON with category, priority, and response.\n\nSubject: {obs.get('subject')}\nBody: {obs.get('body')}"

    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "llama-3.1-8b-instant", 
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # Clean JSON if model adds backticks
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()

        parsed = json.loads(content)
        return {
            "category": parsed.get("category", "internal"),
            "priority": parsed.get("priority", "low"),
            "response": parsed.get("response", "fallback")
        }
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return {"category": "internal", "priority": "low", "response": "error"}

def safe_post(url, payload=None):
    try:
        res = requests.post(url, json=payload, timeout=10) if payload else requests.post(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except:
        return None

def run():
    print("[START] task=email_triage")
    obs = safe_post(f"{API_URL}/reset")
    if not obs: return
    
    action = simple_agent(obs)
    res = safe_post(f"{API_URL}/step", action)
    
    if res:
        reward = res.get("reward", 0)
        print(f"[STEP] step=1 reward={reward}")
        print(f"[END] task=email_triage score={reward} steps=1")

if __name__ == "__main__":
    run()