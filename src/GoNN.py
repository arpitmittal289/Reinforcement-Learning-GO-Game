from torch.utils.data import TensorDataset
import torch
from GoQlearnerPlayer import QLearner
import pickle
import numpy as np
from scipy.special import softmax
# In[4]:

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
    model.cuda()
    outputs = model(test_data.float().cuda())
    #_, predicted = torch.max(outputs.data, dim=1)
    predicted = outputs.data.cpu()
    return predicted

def train_model(model,training_generator,epochs):
    
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for t in range(epochs): 
        model.train()
        model.float()
        count = 0
        for local_batch_review , local_batch_label in training_generator:
            print("Epoch " + str(t) + " stateprocessed " + str(round((count/len(training_generator)*100),2)) + "% done")
            count += 1
            local_batch_review , local_batch_label = local_batch_review.to(device) , local_batch_label.to(device)
            output = model(local_batch_review.float())
            loss = loss_fn(output, local_batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

    
def accuracy_binary_NN(epochs,training_generator) :
    load_pretrained_binary_nn_model = False
    
    if load_pretrained_binary_nn_model :
        # Load PreTrained Binary NN Model
        saved_binary_nn_model = torch.load('qlearner_nn_model')
        qlearner_nn_model.load_state_dict(saved_binary_nn_model)
    else:
        if __name__ == '__main__': 
            train_model(qlearner_nn_model,training_generator,epochs)
    
    torch.save(qlearner_nn_model.state_dict(), "qlearner_nn_model")
    return qlearner_nn_model

# In[14]:

GOQLearnerplayer = QLearner()

with open("GOQLearnerplayer - onlyHistory", "rb") as fp:   # Unpickling
    GOQLearnerplayer = pickle.load(fp)
  
qValDict = GOQLearnerplayer.q_values
