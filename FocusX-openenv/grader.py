def grade(final_state):
    if final_state["tasks_left"] == 0:
        return "SUCCESS "
    else:
        return "FAILED "