# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:55:06 2022

@author: Arpit Mittal
"""

from GoQlearnerPlayer import QLearner
from GoBoard import GO
import pickle
from timeit import default_timer as timer
fullstart = timer()
PLAYER_B = 1
PLAYER_W = 2
    
GOQLearnerplayer = QLearner()

listOfhistoryGames = []
with open("listOfhistoryGames", "rb") as fp:   # Unpickling
    listOfhistoryGames = pickle.load(fp)

with open("GOQLearnerplayer", "rb") as fp:   # Unpickling
    GOQLearnerplayer = pickle.load(fp)

def encode_state(state, player):
        stateCode = ''.join([str(state[i][j]) for i in range(5) for j in range(5)])
        return str(player)+stateCode
    
start = timer()

count = 0
for game in listOfhistoryGames:
    count += 1
    print('Learning Game '+ str(count))
    newGoBoard = GO(5)
    newGoBoard.init_board(5)
    history_states = []
    for seq in game.get_main_sequence() :
        for node in seq:
            move = node.get_move()
            playerLabel = move[0]
            position = move[1]
            if not position:
                continue
            
            player = PLAYER_B
            opp_player = PLAYER_W
            
            if playerLabel == 'w':
                player = PLAYER_W
                opp_player = PLAYER_B
                
            history_states.append((encode_state(newGoBoard.board,player),position))
            newGoBoard.place_chess(position[0], position[1], player)
            newGoBoard.remove_died_pieces(opp_player)
        
    newGoBoard.history_states = history_states
    print("Player Learning")
    GOQLearnerplayer.set_side(PLAYER_B)
    GOQLearnerplayer.learn(newGoBoard)
    print("White Learning")
    GOQLearnerplayer.set_side(PLAYER_W)
    GOQLearnerplayer.learn(newGoBoard)

print("Time Taken for processing Games:", timer()-start) 
pickle.dump(GOQLearnerplayer, open("GOQLearnerplayer", "wb"), protocol=0)
print("Total Time Taken:", timer()-fullstart) 
