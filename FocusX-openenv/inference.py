from env import FocusEnv
import random
import os
import requests   # ✅ correct

# 🔥 Actions
ACTIONS = ["study", "rest", "scroll"]

# 🔥 Q-table
Q = {}

# 🔥 Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# 🔥 Logs
scores = []
actions_taken = []


def get_state_key(state):
    return (
        state["energy"] // 10,
        state["focus"] // 10,
        state["distraction"] // 10
    )


def choose_action(state):
    key = get_state_key(state)

    if key not in Q:
        Q[key] = {a: 0 for a in ACTIONS}

    if random.random() < epsilon:
        return random.choice(ACTIONS)

    return max(Q[key], key=Q[key].get)


def update_q(state, action, reward, next_state):
    key = get_state_key(state)
    next_key = get_state_key(next_state)

    if next_key not in Q:
        Q[next_key] = {a: 0 for a in ACTIONS}

    max_future = max(Q[next_key].values())

    Q[key][action] += alpha * (
        reward + gamma * max_future - Q[key][action]
    )


# ✅ LLM CALL (PROXY SAFE)
def call_llm(state):
    try:
        url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"""
State:
Energy: {state['energy']}
Focus: {state['focus']}
Distraction: {state['distraction']}

Choose best action:
study / rest / scroll

Only output one word.
"""
                }
            ],
            "temperature": 0.3
        }

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        action = text.strip().lower()

        if action not in ACTIONS:
            return "study"

        return action

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", flush=True)
        return "study"


def run():
    env = FocusEnv()

    print("[START] task=FocusX", flush=True)

    total_reward = 0

    # TRAIN
    for _ in range(3):
        state = env.reset()

        for _ in range(10):
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action)

            update_q(state, action, reward, next_state)
            state = next_state

            if done:
                break

    # EVALUATION
    state = env.reset()

    for step in range(10):

        if step == 0:
            action = call_llm(state)   # ✅ mandatory API call
        else:
            action = choose_action(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward
        actions_taken.append(action)
        scores.append(reward)

        print(f"[STEP] step={step+1} reward={reward}", flush=True)

        try:
            advice = env.get_advice()
            print(f"[INFO] action={action} advice={advice}", flush=True)
        except Exception:
            pass

        if done:
            break

    print(f"[END] task=FocusX score={total_reward}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"Actions: {actions_taken}", flush=True)
    print(f"Rewards: {scores}", flush=True)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average Reward: {avg:.2f}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)
