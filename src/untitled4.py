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

D_in, H1, D_out = 26, 26, 25
qlearner_nn_model = QlearnerNN(D_in, H1, D_out)
saved_go_nn_model = torch.load('qlearner_nn_model_midEpoch')
qlearner_nn_model.load_state_dict(saved_go_nn_model)
qlearner_nn_model.cpu()
torch.save(qlearner_nn_model.state_dict(), "qlearner_nn_model_pow_15_mid_cpu", _use_new_zipfile_serialization=False)