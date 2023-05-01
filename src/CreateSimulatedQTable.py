# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:55:06 2022

@author: Arpit Mittal
"""

from GoQlearnerPlayer import QLearner
from GoBoard import GO
import pickle

PLAYER_B = 1
PLAYER_W = 2

GOQLearnerplayer = QLearner()

listOfsimulatedGames = []
with open("listOfsimulatedMiniMaxGames", "rb") as fp:   # Unpickling
    listOfsimulatedGames = pickle.load(fp)

with open("GOQLearnerplayer", "rb") as fp:   # Unpickling
    GOQLearnerplayer = pickle.load(fp)

def encode_state(state, player):
        stateCode = ''.join([str(state[i][j]) for i in range(5) for j in range(5)])
        return str(player)+stateCode
    
    # In[]
count = 0
size = len(listOfsimulatedGames)
start = 370000
end = start + 1000
while end < 400000:
    start = start + 1000
    end = end + 1000
    
    for i in range(start,end,1):
        print('Learning Game '+ str(i))
        game = listOfsimulatedGames[i]
        newGoBoard = GO(5)
        newGoBoard.init_board(5)
        history_states = []
        if game.movesHistory is None:
            continue
        
        for move in game.movesHistory :
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
    
    pickle.dump(GOQLearnerplayer, open("GOQLearnerplayer - HistoryMinimax"+str(end/10000), "wb"), protocol=0)
    print ("-------------------upto " + str(end) + " saved--------------------")