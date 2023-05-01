from torch.utils.data import TensorDataset
import torch
from GoQlearnerPlayer import QLearner
import pickle
import numpy as np
from scipy.special import softmax
from read import readInput

# In[4]:


GOQLearnerplayer = QLearner()

with open("GOQLearnerplayer - onlyHistory", "rb") as fp:   # Unpickling
    GOQLearnerplayer = pickle.load(fp)
  
qValDict = GOQLearnerplayer.q_values
# In[4]:
def encode_state(state, player):
    stateCode = ''.join([str(state[i][j]) for i in range(5) for j in range(5)])
    return str(player) + stateCode


N = 5
piece_type, previous_board, board = readInput(N)
currentStateNN = encode_state(board,piece_type)
print(qValDict[currentStateNN])
