from fastapi import FastAPI
from typing import Optional
from env import CloudScalingEnv
from models import Observation, StepResponse  # ✅ import from models.py
from pydantic import BaseModel

app = FastAPI()

# Global environment
env: Optional[CloudScalingEnv] = None


# -----------------------------
# Request Models (keep here)
# -----------------------------
class ResetRequest(BaseModel):
    difficulty: str = "medium"

class StepRequest(BaseModel):
    action: int


# -----------------------------
# Reset Endpoint
# -----------------------------
@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    global env
    env = CloudScalingEnv(difficulty=req.difficulty)
    return env.reset()


# -----------------------------
# Step Endpoint
# -----------------------------
@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global env

    if env is None:
        return {
            "state": {
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "num_instances": 0,
                "incoming_traffic": 0,
                "time_step": 0
            },
            "reward": 0.0,
            "done": True
        }

    state, reward, done = env.step(req.action)

    return {
        "state": state,
        "reward": float(reward),
        "done": bool(done)
    }