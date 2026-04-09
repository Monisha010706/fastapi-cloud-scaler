from pydantic import BaseModel

class Observation(BaseModel):
    cpu_utilization: float
    memory_utilization: float
    num_instances: int
    incoming_traffic: int
    time_step: int

class Action(BaseModel):
    action: int

class StepResponse(BaseModel):
    state: Observation
    reward: float
    done: bool