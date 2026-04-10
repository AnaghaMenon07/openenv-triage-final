def grade(obs, action, reward, done, info):
    """
    Grader for the urgency_detection task.
    """
    # Success if all fields are present and reward is positive
    has_all = action.get("category") and action.get("priority") and action.get("response")
    success = bool(has_all and reward > 0)
    return success, float(reward)