import torch
import torch.nn as nn
import os
import numpy as np
import read_data

class net3(nn.Module):
    def __init__(self,input_size,nodes,output_size):
        super(net3,self).__init__()
        self.l1 = nn.Linear(input_size, nodes)
        self.activation1 = nn.Sigmoid()
        self.l2 = nn.Linear(nodes, nodes)
        self.activation2 = nn.Sigmoid()
        self.linear_out = nn.Linear(nodes, output_size)
        self.activation_out = nn.Sigmoid()

    def forward(self,x):
        output = self.l1(x)
        output = self.activation1(output)
        output = self.l2(output)
        output = self.activation2(output)
        output = self.linear_out(output)
        output = self.activation_out(output)    
        return output
    
    def train(x_train, y_train, x_valid, y_valid, model, lossfunc, optimizer):
        # Training loop
        train_loss = []
        val_loss = []
        for batch in range(200):
            optimizer.zero_grad()
            outputs = model(torch.tensor(x_train))
            y_train = torch.tensor(y_train)
            loss = lossfunc(outputs.squeeze(1), y_train.squeeze(1))
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                train_loss.append(loss.item())
                
            # Validation step (no gradient updates)
            with torch.no_grad():  # No need to compute gradients for validation
                val_outputs = model(torch.tensor(x_valid))
                val_loss_value = lossfunc(val_outputs.squeeze(1), torch.tensor(y_valid).squeeze(1))
                if batch % 10 == 0:
                    val_loss.append(val_loss_value.item())
                
        # print(train_loss)
        return train_loss, val_loss, model
    
    def test(name, lr, epochs):
        pass
    

    
if __name__ == "__main__":
    data = read_data.read_data("data/23-24-combined.csv")
    x_train = []
    y_train = []
    schools = []
    mapping = {}
    for sample in data:
        x = []
        y = []
        
        for key,val in sample:
            if key == "School":
                schools.append(val)
            elif key == "Placing":
                pass
        x_train.append(x)
        y_train.append(y)
        
    
    node = 9 
    lr = 0.01
    
    model = net3(3,node,1)
    lossfunc = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)