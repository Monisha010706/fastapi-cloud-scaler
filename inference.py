import os
from openai import OpenAI
from env import CloudScalingEnv
from grader import grade

api_base = os.environ.get("API_BASE_URL")
api_key = os.environ.get("API_KEY")

USE_LLM = api_base is not None and api_key is not None

if USE_LLM:
    print("Using Hackathon LLM Proxy", flush=True)
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
else:
    print("Running locally (LLM disabled)", flush=True)
    client = None


# 🔹 LLM Action
def get_llm_action(state):
    if USE_LLM:
        try:
            prompt = f"""
CPU: {state['cpu_utilization']:.2f}
Instances: {state['num_instances']}
Traffic: {state['incoming_traffic']}

Choose:
0 = do nothing
1 = scale up
2 = scale down

Only return number.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            output = response.choices[0].message.content.strip()

            if output in ["0", "1", "2"]:
                return int(output)

        except Exception as e:
            print(f"LLM Error: {e}", flush=True)


    cpu = state["cpu_utilization"]
    if cpu > 0.75:
        return 1
    elif cpu < 0.4:
        return 2
    return 0


def run(difficulty):
    env = CloudScalingEnv(difficulty=difficulty)
    state = env.reset()

    total_reward = 0
    step_count = 0

    print(f"[START] task={difficulty}", flush=True)

    for _ in range(20):
        step_count += 1

        action = get_llm_action(state)
        state, reward, done = env.step(action)

        total_reward += reward

        print(f"[STEP] step={step_count} reward={round(reward, 4)}", flush=True)

        if done:
            break

    score = grade(total_reward)

    print(f"[END] task={difficulty} score={round(score,2)} steps={step_count}", flush=True)

    return {
        "raw_score": round(total_reward, 2),
        "normalized_score": round(score, 2)
    }



if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        run(level)
