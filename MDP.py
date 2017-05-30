import random

class MDP:
    def __init__(self):
        self.know_states = set()
        self.succ = {}
    def register_start_state(self, start_state):
        self.start_state = start_state
        self.know_states.add(start_state)
    def register_actions(self, action_list):
        self.actions = action_list
    def register_operators(self, operators):
        self.operators = operators
    def register_transition_function(self, trans_func):
        self.T = trans_func
    def register_reward_functions(self, reward_func):
        self.R = reward_func
    def state_neighbors(self, state):
        neighbors = self.succ.get(state, False)
        if neighbors == False
            neighbors = [op.apply(state) for op in self.ops if op.is_applicable(state)]