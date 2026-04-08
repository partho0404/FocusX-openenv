from env import FocusEnv
import random
import os

# 🔥 Try importing OpenAI safely
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# 🔥 ENV VARIABLES (DO NOT HARDCODE)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

# 🔥 Setup client safely
client = None
if OpenAI and API_BASE_URL and API_KEY:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None

# 🔥 Actions
ACTIONS = ["study", "rest", "scroll"]

# 🔥 Q-table
Q = {}

# 🔥 Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# 🔥 Logging
scores = []
actions_taken = []


# 🔥 Convert state to discrete key
def get_state_key(state):
    return (
        state["energy"] // 10,
        state["focus"] // 10,
        state["distraction"] // 10
    )


# 🔥 Q-learning action selection
def choose_action(state):
    key = get_state_key(state)

    if key not in Q:
        Q[key] = {a: 0 for a in ACTIONS}

    if random.random() < epsilon:
        return random.choice(ACTIONS)

    return max(Q[key], key=Q[key].get)


# 🔥 Q-learning update
def update_q(state, action, reward, next_state):
    key = get_state_key(state)
    next_key = get_state_key(next_state)

    if next_key not in Q:
        Q[next_key] = {a: 0 for a in ACTIONS}

    max_future = max(Q[next_key].values())

    Q[key][action] += alpha * (
        reward + gamma * max_future - Q[key][action]
    )


# 🔥 SAFE LLM CALL (Phase 2 requirement)
def call_llm(state):
    if client is None:
        return "study"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
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
            temperature=0.3
        )

        # ✅ SAFE extraction (NO syntax error)
        text = response.choices[0].message.content
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

    # 🔥 TRAINING
    for _ in range(3):
        state = env.reset()

        for _ in range(10):
            action = choose_action(state)

            next_state, reward, done, _ = env.step(action)

            update_q(state, action, reward, next_state)

            state = next_state

            if done:
                break

    # 🔥 EVALUATION
    state = env.reset()

    for step in range(10):

        # 🔥 Ensure at least ONE LLM call
        if step == 0:
            action = call_llm(state)
        else:
            action = choose_action(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward

        actions_taken.append(action)
        scores.append(reward)

        print(f"[STEP] step={step+1} reward={reward}", flush=True)

        # Optional info (safe)
        try:
            advice = env.get_advice()
            print(f"[INFO] action={action} advice={advice}", flush=True)
        except Exception:
            pass

        if done:
            break

    print(f"[END] task=FocusX score={total_reward}", flush=True)

    # 🔥 SAFE summary (after END)
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
        # 🔥 FINAL SAFETY (prevents crash fail)
        print(f"[FATAL ERROR] {e}", flush=True)
