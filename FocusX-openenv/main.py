from fastapi import FastAPI
from env import FocusEnv

app = FastAPI()

env = FocusEnv()

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(action: dict):
    act = action.get("action")
    state, reward, done, info = env.step(act)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }
