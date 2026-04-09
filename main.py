from env import FocusEnv
from task import get_tasks

env = FocusEnv()
tasks = get_tasks()

def reset():
    return env.reset()

def step(action):
    return env.step(action)

def state():
    return env._get_state()
