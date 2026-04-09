import os
import requests
import json

API_URL = "http://127.0.0.1:8000"
LLM_URL = "http://127.0.0.1:4000/v1/chat/completions"


def simple_agent(obs):
    prompt = f"""
You are an email triage assistant.

Return ONLY JSON:
{{
  "category": "spam/support/internal",
  "priority": "low/medium/high",
  "response": "your reply"
}}

Email:
Subject: {obs.get("subject", "")}
Body: {obs.get("body", "")}
"""

    try:
        response = requests.post(
            LLM_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            },
            json={
                "model": "gpt-4o-mini",   # ✅ FIXED 
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )

        data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        parsed = json.loads(content)

        return {
            "category": parsed.get("category", "internal"),
            "priority": parsed.get("priority", "low"),
            "response": parsed.get("response", "fallback response")
        }

    except Exception as e:
        print("LLM ERROR:", e)

        return {
            "category": "internal",
            "priority": "low",
            "response": "fallback response"
        }


def safe_post(url, payload=None):
    try:
        if payload:
            res = requests.post(url, json=payload, timeout=5)
        else:
            res = requests.post(url, timeout=5)

        res.raise_for_status()
        return res.json()

    except Exception as e:
        print(f"Request failed at {url}: {e}")
        return None


def run():
    total_score = 0
    step_count = 0

    print("[START] task=email_triage", flush=True)

    obs = safe_post(f"{API_URL}/reset")
    if obs is None:
        return

    done = False

    while not done:
        action = simple_agent(obs)
        res = safe_post(f"{API_URL}/step", action)

        if res is None:
            return

        reward = round(res.get("reward", 0), 2)
        total_score += reward
        step_count += 1

        print(f"[STEP] step={step_count} reward={reward}", flush=True)

        obs = res.get("observation")
        done = res.get("done", True)

        if obs is None:
            break

    print(f"[END] task=email_triage score={round(total_score,2)} steps={step_count}", flush=True)


if __name__ == "__main__":
    run()