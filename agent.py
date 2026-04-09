class Agent:
    def choose_action(self, state):
        if state["distraction"]:
            return "IGNORE"
        elif state["energy"] < 30:
            return "BREAK"
        elif state["focus"] > 40:
            return "STUDY"
        else:
            return "SCROLL"
