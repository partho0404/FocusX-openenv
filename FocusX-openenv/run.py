from env import FocusEnv
from agent import Agent
from task import TASKS
from grader import grade


def display(state):
    focus_bar = "#" * (state["focus"] // 10)
    energy_bar = "#" * (state["energy"] // 10)

    print(f"Focus:  {focus_bar} ({state['focus']})")
    print(f"Energy: {energy_bar} ({state['energy']})")
    print(f"Tasks Left: {state['tasks_left']}")
    print(f"Distraction: {state['distraction']}")
    print("-" * 40)


def run_task(task):
    print(f"\n=== Running {task['name']} Task ===")

    env = FocusEnv(tasks=task["tasks"], max_steps=task["max_steps"])
    agent = Agent()

    done = False
    total_reward = 0
    step_no = 1
    new_state = env.reset()

    while not done:
        print(f"\nStep {step_no}")

        state = env.state()
        action = agent.choose_action(state)
        new_state, reward, done, _info = env.step(action)
        total_reward += reward

        print(f"Action: {action} | Reward: {reward}")
        display(new_state)

        step_no += 1

    result = grade(new_state)

    print("\n--- FINAL RESULT ---")
    print("Result:", result)
    print("Total Score:", total_reward)
    print("=" * 50)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
