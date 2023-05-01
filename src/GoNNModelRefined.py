import torch

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
            print("Epoch " + str(t) + " - " + str(round((count/len(training_generator)*100),2)) + "% done")
            count += 1
            local_batch_review , local_batch_label = local_batch_review.to(device) , local_batch_label.to(device)
            output = model(local_batch_review.float())
            loss = loss_fn(output, local_batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.save(model.state_dict(), "qlearner_nn_model_midEpoch", _use_new_zipfile_serialization=True)

if __name__ == '__main__': 
    power = 15
    D_in, H1, D_out = 26, 26, 25
    qlearner_nn_model = QlearnerNN(D_in, H1, D_out)
    training_generator = torch.load('go_nn_training_generator_pow_'+str(power))
    epochs = 500
    train_model(qlearner_nn_model,training_generator,epochs)
    torch.save(qlearner_nn_model.state_dict(), "qlearner_nn_model_pow_"+str(power))
    qlearner_nn_model.cpu()
    torch.save(qlearner_nn_model.state_dict(), "qlearner_nn_model_pow_"+str(power) +"_cpu", _use_new_zipfile_serialization=True)
