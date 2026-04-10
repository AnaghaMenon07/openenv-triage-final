def grade(obs, action, reward, done, info):
    # Match the 'category' key from your YAML
    success = bool(action.get("category") and reward > 0)
    return success, float(reward)