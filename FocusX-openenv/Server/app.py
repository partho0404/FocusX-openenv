import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from env import FocusEnv
from task import get_tasks

app = FastAPI()

env = FocusEnv()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return get_tasks()

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(action: dict):
    state, reward, done, info = env.step(action.get("action"))
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }
