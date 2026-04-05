import random

class FocusEnv:
    def __init__(self, tasks=3, max_steps=15):
        self.initial_tasks = tasks
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.focus = 70
        self.energy = 60
        self.tasks_left = self.initial_tasks
        self.step_count = 0
        self.distraction = False

        return self.state()

    def state(self):
        return {
            "focus": self.focus,
            "energy": self.energy,
            "tasks_left": self.tasks_left,
            "distraction": self.distraction
        }

    def get_state(self):
        return self.state()

    def step(self, action):
        self.step_count += 1

        # Random distraction
        self.distraction = random.random() < 0.3

        reward = 0

        if action == "STUDY":
            self.energy -= 10
            self.tasks_left -= 1
            reward += 10

        elif action == "SCROLL":
            self.focus -= 15
            reward -= 8

        elif action == "BREAK":
            self.energy += 10
            reward += 2

        elif action == "IGNORE":
            if self.distraction:
                self.focus += 5
                reward += 5

        # Clamp values
        self.focus = max(0, min(100, self.focus))
        self.energy = max(0, min(100, self.energy))

        done = (
            self.tasks_left <= 0
            or self.step_count >= self.max_steps
        )

        info = {
            "step_count": self.step_count
        }

        return self.state(), reward, done, info
