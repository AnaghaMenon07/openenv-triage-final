from env.environment import EmailTriageEnv
from env.models import Action


def simple_agent(observation):
    """
    Rule-based baseline agent
    """
    subject = observation.subject.lower()
    body = observation.body.lower()

    # spam detection
    if "win" in subject or "click" in body or "congratulations" in body:
        return Action(
            category="spam",
            priority="low",
            response="This email appears to be spam and has been flagged accordingly."
        )

    # urgent/production issues
    elif "urgent" in subject or "down" in body or "error" in body or "500" in body:
        return Action(
            category="support",
            priority="high",
            response="We have detected a critical issue. Our team is on it immediately and will resolve this ASAP."
        )

    # refund/billing
    elif "refund" in subject or "broken" in body or "unhappy" in body:
        return Action(
            category="billing",
            priority="high",
            response="We are sorry to hear about your experience. We will process your refund within 2-3 business days."
        )

    # default
    else:
        return Action(
            category="inquiry",
            priority="medium",
            response="Thank you for reaching out. We have received your message and will respond shortly."
        )


def run():
    env = EmailTriageEnv()
    total_score = 0.0
    num_tasks = 3

    print("🚀 Starting baseline evaluation across all 3 tasks...\n")

    for i in range(num_tasks):
        obs = env.reset()

        print(f"📩 Task {i+1} Email:")
        print(f"  Subject: {obs.subject}")
        print(f"  Body: {obs.body}")

        action = simple_agent(obs)

        print(f"\n🤖 Agent Action:")
        print(f"  Category: {action.category}")
        print(f"  Priority: {action.priority}")
        print(f"  Response: {action.response}")

        obs, reward, done, info = env.step(action)

        print(f"\n⭐ Reward: {reward:.2f}")
        total_score += reward
        print("-" * 50)

    avg_score = total_score / num_tasks
    print(f"\n🏁 Finished!")
    print(f"Total Score: {total_score:.2f}")
    print(f"Average Score: {avg_score:.2f}")


if __name__ == "__main__":
    run()