import torch
import torch.nn as nn
import os
import numpy as np
import read_data
from matplotlib import pyplot as plt

class net3(nn.Module):
    def __init__(self,input_size,nodes_1, nodes_2, output_size):
        super(net3,self).__init__()
        self.l1 = nn.Linear(input_size, nodes_1)
        self.dropout1 = nn.Dropout(0.2)
        self.activation1 = nn.ReLU()
        
        # self.l2 = nn.Linear(nodes_1, nodes_2)
        # self.dropout2 = nn.Dropout(0.2)
        # self.activation2 = nn.ReLU()
        
        self.linear_out = nn.Linear(nodes_1, output_size)
        # self.activation_out = nn.Softmax(dim=1)

    def forward(self,x):
        output = self.l1(x)
        output = self.activation1(output)
        output = self.dropout1(output)
        
        # output = self.l2(output)
        # output = self.activation2(output)
        # output = self.dropout2(output)
        
        #crossentropoy loss expects logits so no need to apply activation function
        output = self.linear_out(output) 
        
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
            # print(train_loss)
            
        # # Validation step (no gradient updates)
        # with torch.no_grad():  # No need to compute gradients for validation
        #     val_outputs = model(torch.tensor(x_valid))
        #     val_loss_value = lossfunc(val_outputs.squeeze(1), torch.tensor(y_valid).squeeze(1))
        #     if batch % 10 == 0:
        #         val_loss.append(val_loss_value.item())
            
    return train_loss, model
    
def test(preds, targs):
    correct = 0
    
    for i in range(len(preds)):
        if preds[i] == targs[i]:
            correct += 1
            
    return correct / len(preds)

if __name__ == "__main__":
    data = read_data.read_data("data/all-years-combined.csv")
    x_data = []
    y_data = []
    schools = []
    
    #id to placing
    id_to_placing = [1,2,4,8,16,32,64,68]
    
    #placing to id
    placing_to_id = {
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
        # print(sample)
        for key,val in sample.items():
            if key == "Placing":
                y_data.append(placing_to_id[int(val)])
            elif key != "Rk" and key != "School":
                x.append(float(val))
        x_data.append(x)
        
    x_train = torch.tensor(x_data[:500], dtype=torch.float32)
    x_test = torch.tensor(x_data[500:], dtype=torch.float32)
    
    y_train = torch.tensor(y_data[:500], dtype=torch.long)
    y_test = torch.tensor(y_data[500:], dtype=torch.long)
    
    nodes_1 = 40 
    nodes_2 = 30
    lr = 0.001
    epochs = 25 
    
    model = net3(40,nodes_1, nodes_2, 8)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)
    
    train_losses = []
    for i in range(epochs):
        train_loss, model = train(x_train, y_train, model, lossfunc, optimizer)
        train_losses.append(train_loss)
    # print(train_losses)
    
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
    print(test(predicted_classes, y_test))