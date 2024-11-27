from matplotlib import pyplot as plt
import parse, random, ID3
from forest import RandomForest
  
#Q6
def testRandomForest():
    forest = RandomForest("data/thresholded_output.csv", 7)
    acc = forest.testForestAcc()
    print(acc)

if __name__ == "__main__":
    testRandomForest()