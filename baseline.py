import matplotlib.pyplot as plt
from env import CloudScalingEnv


def run_env(level):
    env = CloudScalingEnv(difficulty=level)
    state = env.reset()

    rewards = []
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

        rewards.append(reward)
        total_reward += reward

    return rewards, total_reward


# Run all difficulty levels
for level in ["easy", "medium", "hard"]:
    rewards, total = run_env(level)

    print(f"{level.upper()} SCORE: {round(total,2)}")

    plt.plot(rewards, label=level)


# Graph settings
plt.title("Cloud Scaling Performance")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()

# Save graph (IMPORTANT for reliability)
plt.savefig("output.png")

print("Graph saved as output.png")

# Optional: also show graph (may or may not open depending on system)
plt.show()