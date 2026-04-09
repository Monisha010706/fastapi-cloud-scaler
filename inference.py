from env import CloudScalingEnv
from grader import grade


def run(difficulty):
    env = CloudScalingEnv(difficulty=difficulty)
    state = env.reset()

    total_reward = 0

    for _ in range(50):
        cpu = state["cpu_utilization"]

        if cpu > 0.75:
            action = 1
        elif cpu < 0.4:
            action = 2
        else:
            action = 0

        state, reward, done = env.step(action)
        total_reward += reward

    normalized_score = grade(total_reward)

    return {
        "raw_score": round(total_reward, 2),
        "normalized_score": round(normalized_score, 2)
    }


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        result = run(level)

        print(f"{level.upper()}:")
        print(f"  Raw Score: {result['raw_score']}")
        print(f"  Normalized Score: {result['normalized_score']}")
        print()