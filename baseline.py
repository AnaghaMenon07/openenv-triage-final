from env.environment import EmailTriageEnv
from env.models import Action

def simple_agent(observation):
    """
    Agent logic that maps the email content to the 'category' field 
    used in the synchronized environment.
    """
    subject = observation.subject.lower()
    body = observation.body.lower()

    # 1. Spam detection (iPhone email)
    if "iphone" in body or "winner" in subject:
        return Action(
            category="spam",
            priority="low",
            response="This looks like a phishing attempt or spam."
        )

    # 2. Support case (Broken order email)
    elif "refund" in subject or "broken" in body:
        # Note: environment.py VALID_CATEGORIES include 'billing' or 'support'
        return Action(
            category="billing",
            priority="medium",
            response="I am sorry to hear your order arrived broken. I will process your refund."
        )

    # 3. Urgency case (Server down email)
    elif "server" in body or "500" in body:
        return Action(
            category="support",
            priority="high",
            response="URGENT: I have notified the engineering team that the database is down."
        )

    # Default fallback
    else:
        return Action(
            category="inquiry",
            priority="low",
            response="Thank you for your email. We will get back to you soon."
        )


def run():
    env = EmailTriageEnv()
    
    # Running 3 iterations to cycle through your new dataset
    for i in range(3):
        print(f"\n--- Task {i+1} ---")
        obs = env.reset()
        
        # obs is now an Observation object (or dict depending on env implementation)
        # Accessing fields for display
        print("📩 Email:")
        print(f"Subject: {obs.subject}")
        print(f"Body: {obs.body}")

        # The agent returns an Action object
        action = simple_agent(obs)

        print("\n🤖 Agent Action:")
        print(f"Category: {action.category}, Priority: {action.priority}")
        print(f"Response: {action.response}")

        # Environment step
        next_obs, reward, done, info = env.step(action)

        print(f"\n⭐ Reward: {reward:.2f}")

    print("\n🏁 Finished testing all 3 tasks!")


if __name__ == "__main__":
    run()