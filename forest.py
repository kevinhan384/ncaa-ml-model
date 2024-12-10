import ID3
import read_data
import random
from collections import defaultdict
import statistics
import numpy as np

class RandomForest():    
    def __init__(self, train, valid, test, num_trees):
        self.test = test
        self.forest = []
        for _ in range(num_trees):
            random.shuffle(train)
            train_data = train[:300]
            tree = ID3.ID3(train_data, 64)
            # pruned = ID3.prune(tree, valid)
            self.forest.append(tree)
        
    def testForestAcc(self):
        correct = 0
        d = defaultdict(dict)

        targs = []
        preds = []
        for example in self.test:
            targs.append(example["Placing"])
            output_counter = defaultdict(int)
            for tree in self.forest:
                output_counter[ID3.evaluate(tree, example)] += 1
            # Get the most common prediction by selecting the key with the max value
            most_common = max(output_counter, key=output_counter.get)
            preds.append(most_common)
            d[example["Placing"]][most_common] = d[example["Placing"]].get(most_common, 0) + 1
            if most_common == example["Placing"]:
                # print(example, "\n", most_common, "\n\n")
                correct += 1
                
        # confusion matrix
        num_classes = 8
        mat = np.zeros((num_classes, num_classes), dtype=int)
        
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
        
        corrects = 0
        targs_classes = [placing_to_id[int(i)] for i in targs]
        preds_classes = [placing_to_id[int(i)] for i in preds]
        
        for i in range(len(preds_classes)):
            if preds_classes[i] == targs_classes[i] or preds_classes[i] == targs_classes[i]-1 or preds_classes[i] == targs_classes[i]+1:
                corrects += 1
        
        print(corrects/len(targs_classes))
        
        
        # Populate the confusion matrix
        for actual, predicted in zip(targs, preds):
            mat[placing_to_id[int(actual)]][placing_to_id[int(predicted)]] += 1
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
    
        
        # res = []

        # for key in d:
        #     l = []
        #     for k in d:
        #         if k in d[key]:
        #             l.append(d[key][k])
        #         else:
        #             l.append(0)

        #     res.append(l)

        # for i in res:
        #     print(i)


        # precisions = []
        # for key in dTP:
        #     if dTP[key] + dFP.get(key, 0) > 0:
        #         precisions.append(dTP[key] / (dTP[key] + dFP.get(key, 0)))

        # precision = statistics.mean(precisions)
        # print(f"Precision: {precision}")

        # recalls = []
        # for key in dTP:
        #     if dTP[key] + dFN.get(key, 0) > 0:
        #         recalls.append(dTP[key] / (dTP[key] + dFN.get(key, 0)))

        # recall = statistics.mean(recalls)
        # print(f"Recall: {recall}")
        
        # forest_acc = correct / len(self.test)
        # return forest_acc

        
    def compare(self):
        tree_acc = ID3.test(self.tree, self.test)
        forest_acc = self.testForestAcc()
        return forest_acc >= tree_acc


    