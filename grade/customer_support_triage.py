def grade(obs, action, reward, done, info):
    """
    Grader for the customer_support_triage task.
    Success if category is provided and reward is positive.
    """
    success = bool(action.get("category") and reward > 0)
    return success, float(reward)