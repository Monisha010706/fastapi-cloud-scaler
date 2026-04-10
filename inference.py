from env import CloudScalingEnv
from grader import grade


def run(difficulty):
    env = CloudScalingEnv(difficulty=difficulty)
    state = env.reset()

    total_reward = 0
    steps = 0


    print(f"[START] task={difficulty}", flush=True)

    for _ in range(50):
        cpu = state["cpu_utilization"]

        if cpu > 0.75:
            action = 1   # scale up
        elif cpu < 0.4:
            action = 2   # scale down
        else:
            action = 0   # do nothing

        state, reward, done = env.step(action)

        total_reward += reward
        steps += 1

        
        print(f"[STEP] step={steps} reward={round(reward, 3)}", flush=True)

        if done:
            break

    normalized_score = grade(total_reward)

    
    print(
        f"[END] task={difficulty} score={round(normalized_score, 3)} steps={steps}",
        flush=True
    )


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        run(level)
