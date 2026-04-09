from env.environment import EmailTriageEnv
from env.models import Action


def simple_agent(observation):
    """
    Dummy agent logic
    """
    subject = observation.subject.lower()
    body = observation.body.lower()

    # spam detection
    if "win" in subject or "click" in body:
        return Action(
            category="spam",
            priority="low",
            response="This looks suspicious."
        )

    # support case
    elif "refund" in subject or "issue" in body:
        return Action(
            category="support",
            priority="high",
            response="Sorry for the issue. We will help resolve it."
        )

    # default internal
    else:
        return Action(
            category="internal",
            priority="medium",
            response="Noted. Will take action accordingly."
        )


def run():
    env = EmailTriageEnv()
    obs = env.reset()

    done = False
    total_score = 0

    while not done:
        print("\n📩 Email:")
        print(f"Subject: {obs.subject}")
        print(f"Body: {obs.body}")

        action = simple_agent(obs)

        print("\n🤖 Agent Action:")
        print(action)

        obs, reward, done, _ = env.step(action)

        print(f"\n⭐ Reward: {reward:.2f}")
        total_score += reward

    print("\n🏁 Finished!")
    print(f"Total Score: {total_score:.2f}")


# 🚨 THIS LINE IS CRITICAL
if __name__ == "__main__":
    run()