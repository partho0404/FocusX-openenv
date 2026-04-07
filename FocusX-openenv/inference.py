from env import FocusEnv

def run():
    env = FocusEnv()
    
    print("[START] task=FocusX", flush=True)

    state = env.reset()
    
    total_reward = 0

    for step in range(5):
        action = "study"  # or any valid action
        state, reward, done, info = env.step(action)
        
        total_reward += reward

        print(f"[STEP] step={step+1} reward={reward}", flush=True)

        if done:
            break

    print(f"[END] task=FocusX score={total_reward}", flush=True)


if __name__ == "__main__":
    run()
