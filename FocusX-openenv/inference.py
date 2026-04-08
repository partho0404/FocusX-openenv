import os
import asyncio
from openai import OpenAI

# ✅ Import your environment
from my_env_v4 import MyEnvV4Env

# =========================
# ENV VARIABLES (MANDATORY)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8

# =========================
# INIT CLIENT (IMPORTANT)
# =========================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# =========================
# LLM ACTION GENERATOR
# =========================
def get_action_from_llm(state):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an RL agent."},
                {"role": "user", "content": f"Current state: {state}. Suggest next action."}
            ],
            temperature=0.3,
        )

        action = response.choices[0].message.content.strip()

        # fallback safety
        if not action:
            return "noop"

        return action

    except Exception as e:
        return "noop"


# =========================
# MAIN EPISODE RUNNER
# =========================
async def run_episode():
    env = MyEnvV4Env()

    state = await env.reset()

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    success = False

    for step in range(1, MAX_STEPS + 1):
        try:
            # ✅ LLM call (MANDATORY)
            action = get_action_from_llm(state)

            result = await env.step(action)

            next_state = result.state
            reward = float(result.reward)
            done = result.done
            error = result.error if result.error else "null"

            rewards.append(f"{reward:.2f}")

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}"
            )

            state = next_state

            if done:
                success = reward > 0
                break

        except Exception as e:
            print(
                f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}"
            )
            break

    # =========================
    # END BLOCK (MANDATORY)
    # =========================
    total_steps = len(rewards)
    score = float(rewards[-1]) if rewards else 0.0

    print(
        f"[END] success={str(success).lower()} steps={total_steps} score={score:.2f} rewards={','.join(rewards)}"
    )

    await env.close()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    asyncio.run(run_episode())
