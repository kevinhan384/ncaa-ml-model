import torch
import torch.nn as nn
import os
import numpy as np
import read_data
from matplotlib import pyplot as plt

class net3(nn.Module):
    def __init__(self,input_size,nodes,output_size):
        super(net3,self).__init__()
        self.l1 = nn.Linear(input_size, nodes)
        self.l2 = nn.Linear(nodes, nodes)
        self.activation2 = nn.ReLU()
        self.linear_out = nn.Linear(nodes, output_size)
        self.activation_out = nn.Softmax(dim=1)

    def forward(self,x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.activation2(output)
        output = self.linear_out(output)
        output = self.activation_out(output)    
        return output
    
def train(x_train, y_train, model, lossfunc, optimizer):
    # Training loop
    train_loss = []
    # val_loss = []
    for batch in range(200):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = lossfunc(outputs, y_train)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())
            
        # # Validation step (no gradient updates)
        # with torch.no_grad():  # No need to compute gradients for validation
        #     val_outputs = model(torch.tensor(x_valid))
        #     val_loss_value = lossfunc(val_outputs.squeeze(1), torch.tensor(y_valid).squeeze(1))
        #     if batch % 10 == 0:
        #         val_loss.append(val_loss_value.item())
            
    # print(train_loss)
    return train_loss, model
    
def test(name, lr, epochs):
    pass

if __name__ == "__main__":
    data = read_data.read_data("data/23-24-combined.csv")
    x_data = []
    y_data = []
    schools = []
    mapping = {
        1:0,
        2:1,
        4:2,
        8:3,
        16:4,
        32:5,
        64:6,
        68:7
    }
    
    for sample in data:
        x = [1.0]
        for key,val in sample.items():
            if key == "School":
                schools.append(val)
            elif key == "Placing":
                y_data.append(mapping[int(val)])
            else:
                if key != "Rk":
                    x.append(float(val))
        x_data.append(x)
        
    x_train = torch.tensor(x_data[:50], dtype=torch.float32)
    x_test = torch.tensor(x_data[50:], dtype=torch.float32)
    train_schools = schools[:50]
    
    y_train = torch.tensor(y_data[:50], dtype=torch.long)
    y_test = torch.tensor(y_data[50:], dtype=torch.long)
    test_schools = schools[50:]
    
    node = 20 
    lr = 0.01
    epochs = 25
    
    model = net3(40,node,8)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    for i in range(epochs):
        train_loss, model = train(x_train, y_train, model, lossfunc, optimizer)
        train_losses.append(train_loss)
    
        #plot losses
    # plt.figure()
    # plt.plot([i for i in range(epochs)], [sum(epoch_loss)/len(epoch_loss) for epoch_loss in train_loss])
    # plt.xlabel("Epochs")
    # plt.ylabel(" Loss")
    # plt.show()
    
        
    outputs = model(x_test)
    _ , predicted_classes = torch.max(outputs, dim=1)
    print(y_test)
    print(predicted_classes)