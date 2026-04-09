from fastapi import FastAPI
from pydantic import BaseModel
from env import FocusEnv
from task import get_tasks

app = FastAPI()
env = FocusEnv()

class ActionRequest(BaseModel):
    action: str

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def tasks():
    return get_tasks()

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.post("/step")
def step(request: ActionRequest):
    state, reward, done, info = env.step(request.action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }
