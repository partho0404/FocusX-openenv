import random

class FocusEnv:

    def __init__(self):
        self.user_type = random.choice(["lazy", "focused", "distracted"])
        self.reset()

    def reset(self):
        self.energy = 100
        self.focus = 50
        self.distraction = 30
        self.streak = 0
        return self._get_state()

    def step(self, action):
        reward = 0

        if action == "study":
            self.focus += 10
            self.energy -= 10
            self.distraction -= 5
            self.streak += 1
            reward += 10 + self.streak * 2

        elif action == "scroll":
            self.focus -= 10
            self.distraction += 15
            self.energy -= 5
            self.streak = 0
            reward -= 10

        elif action == "rest":
            self.energy += 15
            self.distraction -= 5
            reward += 5

        # Personalization
        if self.user_type == "lazy":
            reward += 5 if action == "study" else -5

        if self.user_type == "distracted" and action == "scroll":
            reward -= 15

        # Clamp values
        self.energy = max(0, min(100, self.energy))
        self.focus = max(0, min(100, self.focus))
        self.distraction = max(0, min(100, self.distraction))

        done = self.energy == 0

        return self._get_state(), reward, done, {}

    # 🔥 REQUIRED BY OPENENV
    def state(self):
        return self._get_state()

    def _get_state(self):
        return {
            "energy": self.energy,
            "focus": self.focus,
            "distraction": self.distraction
        }

    def get_advice(self):
        if self.distraction > 70:
            return "Reduce distractions"
        if self.energy < 30:
            return "Take a break"
        return "Stay focused"
