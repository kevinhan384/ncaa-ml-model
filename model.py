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

    def forward(self,x):
        output = self.l1(x)
        output = self.activation1(output)
        output = self.dropout1(output)
        
        # output = self.l2(output)
        # output = self.activation2(output)
        # output = self.dropout2(output)
        
        #crossentropy loss expects logits so no need to apply activation function
        output = self.linear_out(output) 
        
        return output
    
def train(x_train, y_train, model, lossfunc, optimizer):
    # Training loop
    train_loss = []
    for batch in range(200):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = lossfunc(outputs, y_train)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())
            
    return train_loss, model
    
def test(preds, targs):
    correct = 0
    
    for i in range(len(preds)):
        if preds[i] == targs[i] or preds[i] == targs[i]-1 or preds[i] == targs[i]+1:
            correct += 1

    # confusion matrix
    num_classes = 8
    mat = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate the confusion matrix
    for actual, predicted in zip(targs, preds):
        mat[actual][predicted] += 1
    
    print(mat)
    
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(mat)):
        TP += mat[i][i]  # True positives for class i
        FP += sum(mat[j][i] for j in range(num_classes)) - mat[i][i]  # False positives for class i
        FN += sum(mat[i][j] for j in range(num_classes)) - mat[i][i]  # False negatives for class i
    
    prec = TP / (TP+FP)
    rec = TP / (TP + FN)
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0
    
    print(f"F1: {f1} ")
    
    print(correct / len(preds))

def cross_validate():
    years = ['14-15','15-16','16-17','17-18','18-19','20-21', '21-22', '22-23', '23-24']
    
    n_splits = len(years)
    
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
        
    for i in range(n_splits):
        train_years = years[0:i] + years[i+1:]
        test_year = years[i]
        
        data_train = []
        for year in train_years:
            data = read_data.read_data(f"data/{year}-combined.csv")
            data_train += data
            
        x_train = []
        y_train = []
        schools = []
        
        for sample in data_train:
            x = [1.0]
            # print(sample)
            for key,val in sample.items():
                if key == "Placing":
                    y_train.append(placing_to_id[int(val)])
                elif key != "Rk" and key != "School":
                    x.append(float(val))
            x_train.append(x)
            
        data_test = read_data.read_data(f"data/{test_year}-combined.csv")

        x_test = []
        y_test = []
        schools = []
        
        for sample in data_test:
            x = [1.0]
            for key,val in sample.items():
                if key == "Placing":
                    y_test.append(placing_to_id[int(val)])
                elif key == "School":
                    schools.append(val)
                elif key != "Rk":
                    x.append(float(val))
            x_test.append(x)
            
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        nodes_1 = 40 
        nodes_2 = 30
        lr = 0.001
        epochs = 25 
        
        model = net3(41,nodes_1, nodes_2, 8)
        lossfunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)
        
        train_losses = []
        for i in range(epochs):
            train_loss, model = train(x_train, y_train, model, lossfunc, optimizer)
            train_losses.append(train_loss)
            
        outputs = model(x_test)
        _ , predicted_classes = torch.max(outputs, dim=1)

        for i,val in enumerate(y_test):
            print(f"{schools[i]} - Predicted {id_to_placing[predicted_classes[i]]}, Actual {id_to_placing[val]} ")        
        
        test(predicted_classes, y_test)
        
        print("\n\n")
        
        

if __name__ == "__main__":
    cross_validate()