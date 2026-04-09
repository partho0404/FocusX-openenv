import os
from openai import OpenAI
from env import FocusEnv

# =========================
# ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASK_NAME = "focus_task_1"
BENCHMARK = "focus-env"
MAX_STEPS = 8

# =========================
# CLIENT
# =========================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

ACTIONS = ["study", "rest", "scroll"]

# =========================
# LLM ACTION
# =========================
def get_action_from_llm(state):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Choose one action: study, rest, scroll."
                },
                {
                    "role": "user",
                    "content": f"State: {state}"
                }
            ],
            temperature=0.3
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in ACTIONS:
            return "study"

        return action

    except Exception:
        return "study"

# =========================
# RUN
# =========================
def run():
    env = FocusEnv()
    state = env.reset()

    rewards = []
    success = False

    # START
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    for step in range(1, MAX_STEPS + 1):
        try:
            action = get_action_from_llm(state)

            next_state, reward, done, info = env.step(action)

            error = "null"
            if isinstance(info, dict):
                error = info.get("error", "null")

            rewards.append(f"{float(reward):.2f}")

            # STEP
            print(
                f"[STEP] step={step} action={action} reward={float(reward):.2f} done={str(done).lower()} error={error}",
                flush=True
            )

            state = next_state

            if done:
                success = reward > 0
                break

        except Exception as e:
            print(
                f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}",
                flush=True
            )
            break

    # SCORE NORMALIZATION
    if rewards:
        avg_reward = sum([float(r) for r in rewards]) / len(rewards)
        score = (avg_reward + 10) / 20
        score = max(0.01, min(0.99, score))
    else:
        score = 0.5

    # END
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.2f} rewards={','.join(rewards)}",
        flush=True
    )

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()
