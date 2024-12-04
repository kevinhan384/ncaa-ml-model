import ID3
import read_data
import random
from collections import defaultdict
import statistics

class RandomForest():
    def __init__(self, filename, num_trees):
        data = read_data.read_data(filename)
        self.test = data[520:600]
        valid = data[450:520]
        self.forest = []
        for _ in range(num_trees):
            random.shuffle(data)
            train = data[0:450]
            tree = ID3.ID3(train, 64)
            pruned = ID3.prune(tree, valid)
            self.forest.append(pruned)
        
    def testForestAcc(self):
        correct = 0
        d = defaultdict(dict)

        for example in self.test:
            output_counter = defaultdict(int)
            for tree in self.forest:
                output_counter[ID3.evaluate(tree, example)] += 1
            # Get the most common prediction by selecting the key with the max value
            most_common = max(output_counter, key=output_counter.get)
            d[example["Placing"]][most_common] = d[example["Placing"]].get(most_common, 0) + 1
            if most_common == example["Placing"]:
                # print(example, "\n", most_common, "\n\n")
                correct += 1
        forest_acc = correct / len(self.test)

        res = []

        for key in d:
            l = []
            for k in d:
                if k in d[key]:
                    l.append(d[key][k])
                else:
                    l.append(0)

            res.append(l)

        for i in res:
            print(i)


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

        return forest_acc

        
    def compare(self):
        tree_acc = ID3.test(self.tree, self.test)
        forest_acc = self.testForestAcc()
        return forest_acc >= tree_acc


    