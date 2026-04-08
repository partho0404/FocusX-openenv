def grade_focus(trajectory):
    try:
        if not trajectory or len(trajectory) == 0:
            return 0.5
        rewards = []
        for step in trajectory:
            if "reward" in step:
                rewards.append(step["reward"])
        if not rewards:
            return 0.5
        avg = sum(rewards) / len(rewards)
        score = (avg + 10) / 20
        score = max(0.01, min(0.99, float(score)))
        return score
    except Exception:
        return 0.5

def grade(trajectory):
    return grade_focus(trajectory)
