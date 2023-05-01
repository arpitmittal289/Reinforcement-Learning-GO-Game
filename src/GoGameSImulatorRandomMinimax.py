# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 01:43:32 2022

@author: Arpit Mittal
"""
import random
import math
from copy import deepcopy
import pickle

def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
    	res = "PASS"
    else:
	    res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
	with open(path, 'w') as f:
		f.write("PASS")

def writeNextInput(piece_type, previous_board, board, path="input.txt"):
	res = ""
	res += str(piece_type) + "\n"
	for item in previous_board:
		res += "".join([str(x) for x in item])
		res += "\n"
        
	for item in board:
		res += "".join([str(x) for x in item])
		res += "\n"

	with open(path, 'w') as f:
		f.write(res[:-1]);


class GO:
    movesHistory = []
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False
    
    def find_liberty_value2(self, i, j, piece):
        return 1
    
    def find_liberty_value(self, i, j, piece):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        count = 0
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    count += 1
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return count
    
    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt          

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''        

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi: return 1
        elif cnt_1 < cnt_2 + self.komi: return 2
        else: return 0
        
    def play(self, player1, player2, verbose=False):
        '''
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        self.init_board(self.size)
        # Print input hints and error message if there is a manual player
        if player1.type == 'manual' or player2.type == 'manual':
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print('X stands for black chess, O stands for white chess.')
            self.visualize_board()
        
        verbose = self.verbose
        # Game starts!
        while 1:
            piece_type = 1 if self.X_move else 2

            # Judge if the game should end
            if self.game_end(piece_type):       
                result = self.judge_winner()
                if verbose:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            # Game continues
            if piece_type == 1: action = player1.get_input(self, piece_type)
            else: action = player2.get_input(self, piece_type)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board() 
                    continue

                self.died_pieces = self.remove_died_pieces(3 - piece_type) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board() # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move # Players take turn

class MyGoPlayer():
    
    center_control = [[(2,2)], [(1,2), (3,2), (2,1), (2,3)], [(1,1), (1,3), (3,1), (3,3)]]
     
    global_piece_type = -1
    
    def __init__(self, piece_type):
        self.type = 'random'
        self.global_piece_type = piece_type

    
    def get_random_input(self, go, piece_type):
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    possible_placements.append((i,j))
        
        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)
        
    def get_minmax_input(self, go, piece_type):
        max_var = 0
        max_pos = (-1,-1)
        opp_piece = 2 if piece_type == 1 else 1
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    go.board[i][j] = piece_type
                    var = len(go.find_died_pieces(opp_piece))
                    if var > max_var:
                        max_var = var
                        max_pos = (i,j)
                    possible_placements.append((i,j))
                    go.board[i][j] = 0
        
        if not possible_placements:
            return "PASS"
        elif max_var > 0:
            return max_pos
        else:
            if self.global_piece_type == 1:
                for preflist in self.center_control:
                    currValidList = []
                    for pos in preflist:
                        if go.valid_place_check(pos[0], pos[1], piece_type, test_check = True):
                            currValidList.append(pos)
                    
                    if len(currValidList) > 0:
                        return random.choice(currValidList)

            return self.minimax(go, 2, -float('inf'), float('inf'), True, piece_type)[0]
    
    def evaluation_func(self, go):
        
        # print("PIECE TYPE " + str(self.global_piece_type))
        opp_piece = 2 if self.global_piece_type == 1 else 1
        
        my_piece_count = 0
        ta_piece_count = 0
        
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == self.global_piece_type:
                    my_piece_count+=go.find_liberty_value(i, j, self.global_piece_type)
                if go.board[i][j] == opp_piece:
                    ta_piece_count+=go.find_liberty_value(i, j, opp_piece)
        
        return my_piece_count - ta_piece_count

    def get_valid_places(self, go, piece_type):
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    possible_placements.append((i,j))
                    
        return possible_placements

    def minimax(self, go, depth, alpha, beta, max_player, piece_type):
        
        # print("DEPTH " + str(depth))
        opp_piece = 2 if piece_type == 1 else 1
        if depth == 0:
            return None, self.evaluation_func(go)
    
        branches = self.get_valid_places(go, piece_type)
        if len(branches) == 0:
            return None, self.evaluation_func(go)
        
        move = branches[0]
        
        if max_player:
            max_score = -float('inf')        
            for branch in branches:
                next_go = go.copy_board()
                next_go.place_chess(branch[0], branch[1], piece_type)
                next_go.remove_died_pieces(opp_piece)
                current_score = self.minimax(next_go, depth - 1, alpha, beta, False, opp_piece)[1]
                if current_score > max_score:
                    max_score = current_score
                    move = branch
                alpha = max(alpha, current_score)
                if beta <= alpha:
                    break
            return move, max_score
    
        else:
            min_score = float('inf')
            for branch in branches:
                next_go = go.copy_board()
                next_go.place_chess(branch[0], branch[1], piece_type)
                next_go.remove_died_pieces(opp_piece)
                current_score = self.minimax(next_go, depth - 1, alpha, beta, True, opp_piece)[1]
                if current_score < min_score:
                    min_score = current_score
                    move = branch
                beta = min(beta, current_score)
                if beta <= alpha:
                    break
            return move, min_score
import torch
import operator

class QlearnerNN(torch.nn.Module):
    def __init__(self, D_in, H1, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.output = torch.nn.Linear(H1, D_out)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.outputShape = D_out
        self.weight_init()
        
    def weight_init(self):
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.ones_(self.linear1.bias)

    def forward(self, x):
        h1 = self.linear1(x).clamp(min=0)
        y_pred = self.output(h1)
        y_pred = self.softmax(y_pred)
        return y_pred

def predict(model,test_data):
    model.eval()
    model.float()
    outputs = model(test_data.float())
    #_, predicted = torch.max(outputs.data, dim=1)
    predicted = outputs.data
    return predicted

def encode_state(state, player):
        stateCode = [state[i][j] for i in range(5) for j in range(5)]
        finalStateCode = []
        finalStateCode.append(player)
        finalStateCode.extend(stateCode)
        return finalStateCode

D_in, H1, D_out = 26, 26, 25
go_nn_model = QlearnerNN(D_in, H1, D_out)
saved_go_nn_model = torch.load('qlearner_nn_model_pow_10_cpu')
go_nn_model.load_state_dict(saved_go_nn_model)
    
def get_qlearn_input(go, piece_type):
    possible_placements = []
    
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check(i, j, piece_type, test_check = True):
                possible_placements.append((i,j))
    
    if len(possible_placements) == 0:
        return "PASS"
    
    currentStateNN = encode_state(go.board,piece_type)
    output = predict(go_nn_model,torch.tensor(currentStateNN))
    
    output2d = output.reshape(5,5).numpy()
    position_qval_map = {}
    for i in range(len(output2d)):
        for j in range(len(output2d[0])):
            position_qval_map[(i,j)] = output2d[i][j]
        
    sorted_map = sorted(position_qval_map.items(), key=operator.itemgetter(1))
    sorted_map.reverse()
    
    finalMove = "PASS"
    for optimalMove in sorted_map:
        move = optimalMove[0]
        if move in possible_placements:
            finalMove = move
            break
            
    return finalMove

def makeMove(piece_type, previous_board, board, isqlearn):
    N = 5
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyGoPlayer(piece_type)
    if isqlearn:
        action = get_qlearn_input(go, piece_type)
    else:
        #action = player.get_random_input(go, piece_type)
        action = player.get_minmax_input(go, piece_type)
    return action
    

def moveplayer(startGO,player,label,listOfMoves, isqlearn):
    oppPiece = PLAYER_W
    if player == PLAYER_W:
        oppPiece = PLAYER_B
    movePlayed = makeMove(player, startGO.previous_board, startGO.board, isqlearn)
    if movePlayed == 'PASS':
        return True
    startGO.place_chess(movePlayed[0], movePlayed[1], player)
    startGO.remove_died_pieces(oppPiece)
    listOfMoves.append((label,movePlayed))
    if startGO.game_end(PLAYER_B):
        return True
    return False

PLAYER_B = 1
PLAYER_W = 2
listOfsimulatedGames = []

def playRandomGame(playerisWhite):   
    startGO = GO(5)
    startGO.init_board(5)
    listOfMoves = []
    
    isBQlearn = not playerisWhite
    isWQlearn = playerisWhite
    
    while True:
        if moveplayer(startGO,PLAYER_B, 'b',listOfMoves,isBQlearn):
            break
        if moveplayer(startGO,PLAYER_W, 'w',listOfMoves,isWQlearn):
            break
        
    startGO.movesHistory = listOfMoves
    return startGO

    # In[]
if __name__ == "__main__":
    count = 0
    for i in range(10):
        playerisWhite = True
        player = 2 if playerisWhite else 1
        game = playRandomGame(playerisWhite)
        if(game.judge_winner() == player):
            count+=1
    print(count)
