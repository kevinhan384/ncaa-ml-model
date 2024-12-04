from forest import RandomForest
  
#Q6
def testRandomForest():
    forest = RandomForest("data/all-years-thresholded.csv", 7)
    acc = forest.testForestAcc()
    print(acc)

if __name__ == "__main__":
    testRandomForest()