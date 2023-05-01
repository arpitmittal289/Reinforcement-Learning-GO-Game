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

def get_qlearn_input(go, piece_type):
    possible_placements = []
    
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check(i, j, piece_type, test_check = True):
                possible_placements.append((i,j))
    
    if len(possible_placements) == 0:
        return "PASS"
    
    D_in, H1, D_out = 26, 26, 25
    go_nn_model = QlearnerNN(D_in, H1, D_out)
    saved_go_nn_model = torch.load('qlearner_nn_model_cpu')
    go_nn_model.load_state_dict(saved_go_nn_model)
    torch.save(go_nn_model.state_dict(), "qlearner_nn_model_cpu", _use_new_zipfile_serialization=False)
    
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
    
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    action = get_qlearn_input(go, piece_type)
    print(action)
    writeOutput(action)
    
    
