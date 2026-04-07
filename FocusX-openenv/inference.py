from env import FocusEnv
import random

# 🔥 Actions
ACTIONS = ["study", "rest", "scroll"]

# 🔥 Q-table
Q = {}

# 🔥 Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# 🔥 Visualization storage
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

    # ε-greedy exploration
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


def run():
    env = FocusEnv()

    print("[START] task=FocusX", flush=True)

    total_reward = 0

    # 🔥 TRAINING PHASE
    for episode in range(3):
        state = env.reset()

        for step in range(10):
            action = choose_action(state)

            next_state, reward, done, _ = env.step(action)

            update_q(state, action, reward, next_state)

            state = next_state

            if done:
                break

    # 🔥 EVALUATION PHASE
    state = env.reset()

    for step in range(10):
        action = choose_action(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward

        # 🔥 Store for visualization
        actions_taken.append(action)
        scores.append(reward)

        print(f"[STEP] step={step+1} reward={reward}", flush=True)

        # 🔥 AI Advice (extra intelligence)
        advice = env.get_advice()
        print(f"[INFO] action={action} advice={advice}", flush=True)

        if done:
            break

    print(f"[END] task=FocusX score={total_reward}", flush=True)

    # 🔥 VISUAL SUMMARY (AFTER END → safe)
    print("\n=== VISUAL SUMMARY ===", flush=True)
    print(f"Total Steps: {len(actions_taken)}", flush=True)
    print(f"Actions Taken: {actions_taken}", flush=True)
    print(f"Rewards: {scores}", flush=True)

    if scores:
        print(f"Average Reward: {sum(scores)/len(scores):.2f}", flush=True)
        print(f"Max Reward: {max(scores)}", flush=True)
        print(f"Min Reward: {min(scores)}", flush=True)


if __name__ == "__main__":
    run()
