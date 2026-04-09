import random

random.seed(42)

class CloudScalingEnv:
    def __init__(self, max_instances=10, difficulty="medium"):
        self.max_instances = max_instances
        self.difficulty = difficulty
        self.reset()

    def reset(self):
        self.cpu = 0.5
        self.memory = 0.5
        self.instances = 3
        self.time = 0

        # Difficulty-based traffic
        if self.difficulty == "easy":
            self.traffic = 80
        elif self.difficulty == "medium":
            self.traffic = random.randint(50, 150)
        else:
            self.traffic = random.randint(100, 200)

        return self.state()

    def state(self):
        return {
            "cpu_utilization": float(self.cpu),
            "memory_utilization": float(self.memory),
            "num_instances": int(self.instances),
            "incoming_traffic": int(self.traffic),
            "time_step": int(self.time)
        }

    def step(self, action):
        # Actions
        if action == 1 and self.instances < self.max_instances:
            self.instances += 1
        elif action == 2 and self.instances > 1:
            self.instances -= 1

        # Traffic changes based on difficulty
        if self.difficulty == "easy":
            self.traffic += random.randint(-5, 5)
        elif self.difficulty == "medium":
            self.traffic += random.randint(-20, 20)
        else:
            self.traffic += random.randint(-50, 50)

        self.traffic = max(10, self.traffic)

        # Update utilization
        self.cpu = min(1.0, self.traffic / (self.instances * 100))
        self.memory = self.cpu * 0.8

        # Reward function
        ideal_cpu = 0.6
        reward = 1 - abs(self.cpu - ideal_cpu)
        reward -= 0.05 * self.instances

        if self.cpu > 0.9:
            reward -= 1.5

        self.time += 1
        done = self.time >= 50

        return self.state(), float(reward), bool(done)