from env import FocusEnv

env = FocusEnv()

def run_inference():
    state = env.reset()
    
    results = []
    
    done = False
    while not done:
        action = "study"  # simple baseline policy
        state, reward, done, info = env.step(action)
        
        results.append({
            "state": state,
            "reward": reward,
            "done": done
        })
    
    return results


if __name__ == "__main__":
    output = run_inference()
    print(output)
