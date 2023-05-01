import numpy as np
from GoBoard import GO
WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0
PLAYER_B = 1
PLAYER_W = 2
class QLearner:

    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, side=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        # self.state = ?

    def set_side(self, side):
        self.side = side

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5, 5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def learn(self, board):
        """ when games ended, this method will be called to update the qvalues
        """
        if board.judge_winner() == 0:
            reward = DRAW_REWARD
        elif board.judge_winner() == self.side:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD

        history_states = board.history_states
        history_states.reverse()
        
        max_q_value = -1.0
        for hist in history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)

