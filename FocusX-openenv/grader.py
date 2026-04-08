def grade_focus(trajectory):
    try:
        if not trajectory or len(trajectory) == 0:
            return 0.5

        rewards = [step.get("reward", 0) for step in trajectory]

        if len(rewards) == 0:
            return 0.5

        avg_reward = sum(rewards) / len(rewards)

        # normalize (-10 to +10 → 0 to 1)
        score = (avg_reward + 10) / 20

        # STRICT bounds (IMPORTANT)
        if score <= 0:
            score = 0.01
        elif score >= 1:
            score = 0.99

        return float(score)

    except Exception:
        return 0.5
