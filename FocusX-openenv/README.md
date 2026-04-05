Project: FocusX – AI Productivity Environment

Overview
     FocusX is a small simulation project that models productivity as a decision-making problem. An agent must choose actions that help complete tasks while managing internal state such as focus, energy, and distraction.
     This project is designed as a simple environment for experimenting with rule-based agents and as a starting point for reinforcement learning ideas.


Features

     Dynamic distraction system
     Multi-level tasks (Easy → Hard)
     Reward-based learning environment
     Rule-based agent (RL-compatible design)

Objective:- 
     Build or improve an agent that can complete all assigned tasks within the allowed number of steps.
     The agent should make decisions based on the current environment state and aim to maximize reward while finishing the task list efficiently.

Environment State

     At each step, the environment provides the agent with the following state values:

     focus: Current focus level of the agent
     energy: Current energy level of the agent
     tasks_left: Number of unfinished tasks
     distraction: Whether a distraction is currently present

Initial environment values:

     focus = 70
     energy = 60
     tasks_left = tasks configured for the level
     distraction = False

Available Actions
     The agent can choose one of the following actions:

     STUDY: Work on a task
     SCROLL: Waste time and lose focus
     BREAK: Recover energy
     IGNORE: Try to resist distraction

Task Design

     Easy: Complete 2 tasks
     Medium: Handle distractions
     Hard: Optimize performance under constraints

Reward Logic
     Task completion: +10
     Distraction: -8
     Focus control: +5
     Energy recovery: +2

After every action:

     focus is clamped between `0` and `100`
     energy is clamped between `0` and `100`
     the episode ends if all tasks are completed or the step limit is reached

Task Levels
     The project currently includes three difficulty levels:

     - `Easy`: `2` tasks, `10` max steps
     - `Medium`: `3` tasks, `12` max steps
     - `Hard`: `4` tasks, `15` max steps

Current Agent Strategy
     The default agent in agent.py uses simple rule-based logic:

     - If distracted, choose `IGNORE`
     - If energy is below `30`, choose `BREAK`
     - If focus is above `40`, choose `STUDY`
     - Otherwise, choose `SCROLL`

This makes the project easy to understand and provides a baseline for future improvements.

Grading Logic
     The current grading rule is intentionally simple:

     - `SUCCESS` if `tasks_left == 0`
     - `FAILED` otherwise

     This means the final result depends only on whether all tasks were completed, not on how efficiently the agent performed.

 Project Files
     env.py: Defines the productivity environment and state transitions
     agent.py: Contains the agent decision logic
     task.py: Stores task difficulty settings
     grader.py: Evaluates final success or failure
     run.py: Runs all task levels and prints the results

How to Run
     Run the project with:

     bash
     python run.py



Why This Project Matters

     FocusX is a good beginner-friendly AI mini-project because it demonstrates:

     - state-based decision making
     - reward-driven behavior
     - simple environment-agent interaction
     - a clear path from rule-based logic to reinforcement learning

Limitations
     The current version is intentionally simple and has room for improvement:

     - the agent uses fixed rules instead of learning
     - grading does not consider score or efficiency
     - distraction handling is basic
     - focus and energy dynamics can be made more realistic

Future Improvements
     Possible extensions for the project include:

     - Q-learning or deep reinforcement learning integration
     - improved reward shaping
     - smarter task difficulty scaling
     - better distraction modeling
     - analytics for agent performance across multiple runs

 Summary
     FocusX is a compact AI assignment that simulates productivity management through an environment-agent loop. It is useful for demonstrating core AI concepts in a simple, readable, and extendable format.