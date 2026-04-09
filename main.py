from fastapi import FastAPI
from pydantic import BaseModel
from env import FocusEnv
from task import get_tasks
import uvicorn

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

@app.post("/reset")        # ← must be POST
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")         # ← must be POST
def step(request: ActionRequest):
    state, reward, done, info = env.step(request.action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
