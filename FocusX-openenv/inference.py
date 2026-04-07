from env import FocusEnv
import random
import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

ACTIONS = ["study", "rest", "scroll"]

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

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

def call_llm(state):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are a productivity assistant.

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

        action = response.choices[0].message.content.strip().
