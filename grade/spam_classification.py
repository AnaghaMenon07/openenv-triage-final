def grade(obs, action, reward, done, info):
    """
    Grader for the spam_classification task.
    Success if both category and priority are present and reward is positive.
    """
    success = bool(action.get("category") and action.get("priority") and reward > 0)
    return success, float(reward)