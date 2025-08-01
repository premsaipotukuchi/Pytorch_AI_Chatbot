import torch
import torch.nn as nn



class NeuralNet(nn.Module):
    def __len__(self,input_size,hidden_size,num_class):
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_class)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.l1(x)
        out =  self.relu(x)
        out = self.l2(x)
        out = self.relu(x)
        out = self.l3(x)
        out = self.relu(x)

        return out

