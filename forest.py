import ID3
import read_data
import random
from collections import defaultdict

class RandomForest():
    def __init__(self, filename, num_trees):
        data = read_data.read_data(filename)
        self.test = data[55:68]
        valid = data[48:55]
        self.forest = []
        for _ in range(num_trees):
            random.shuffle(data)
            train = data[0:48]
            tree = ID3.ID3(train, 64)
            pruned = ID3.prune(tree, valid)
            self.forest.append(pruned)
        
    def testForestAcc(self):
        correct = 0
        for example in self.test:
            output_counter = defaultdict(int)
            for tree in self.forest:
                output_counter[ID3.evaluate(tree, example)] += 1
            # Get the most common prediction by selecting the key with the max value
            most_common = max(output_counter, key=output_counter.get)
            if most_common == example["Placing"]:
                print(example, "\n", most_common, "\n\n")
                correct += 1
        forest_acc = correct / len(self.test)
        return forest_acc
        
    def compare(self):
        tree_acc = ID3.test(self.tree, self.test)
        forest_acc = self.testForestAcc()
        return forest_acc >= tree_acc


    