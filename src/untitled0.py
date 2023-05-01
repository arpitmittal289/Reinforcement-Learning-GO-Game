import random
from read import readInput
from write import writeOutput
from host_copy import GO
import torch
import math
import operator

class QlearnerNN(torch.nn.Module):
    def __init__(self, D_in, H1, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.output = torch.nn.Linear(H1, D_out)
        self.softmax = torch.nn.Softmax()
        self.outputShape = D_out
        self.weight_init()
        
    def weight_init(self):
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.ones_(self.linear1.bias)

    def forward(self, x):
        x= x.cuda()
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
        stateCode = ''.join([str(state[i][j]) for i in range(5) for j in range(5)])
        encodedState =  str(player)+stateCode
        return[eval(i) for i in [*encodedState]

               
def get_input(self, go, piece_type):
    possible_placements = []
    
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check(i, j, piece_type, test_check = True):
                possible_placements.append((i,j))
    
    if len(possible_placements) == 0:
        return "PASS"
    
    go_nn_model = QlearnerNN(D_in, H1, D_out)
    saved_go_nn_model = torch.load('qlearner_nn_model')
    go_nn_model.load_state_dict(saved_go_nn_model)
    
    currentStateNN = encode_state(go.board,piece_type)
    output = predict(go_nn_model,torch.tensor(currentStateNN))
    
    output2d = output.reshape(5,5)
    
    position_qval_map = {}
    for i in len(output2d):
        for j in len(output2d[0]):
            position_qval_map[str(i)+","+str(j)] = output2d[i][j]
        
    sorted_map = sorted(position_qval_map.items(), key=operator.itemgetter(1))
    sorted_map.reverse()
    
    finalMove = (3,3)
    for optimalMove in sorted_map:
        move = optimalMove[0]
        if move in possible_placements:
            finalMove = move
            
    return finalMove
    
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    action = get_input(go, piece_type)
    writeOutput(action)
    
    
