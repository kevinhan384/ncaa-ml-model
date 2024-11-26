from matplotlib import pyplot as plt
import parse, random, ID3
from forest import RandomForest
  
#Q4
def learning_curves():
    data = parse.parse("house_votes_84.data")
    xPoints = []
    yPointsWithPruning = []
    yPointsWithoutPruning = []

    for training_size in range(10, 301):
        withPruning = []
        withoutPruning = []
        subData = data[:training_size + 1]

        for _ in range(100):
            random.shuffle(subData)
            train = subData[:len(subData)//2]
            valid = subData[len(subData)//2:3*len(subData)//4]
            test = subData[3*len(subData)//4:]

            # Without pruning
            tree = ID3.ID3(train + valid, "1")
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)

            # With pruning
            tree = ID3.ID3(train, "1")
            tree = ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            withPruning.append(acc)

        avgWithPruning = sum(withPruning) / len(withPruning)
        avgWithoutPruning = sum(withoutPruning) / len(withoutPruning)

        xPoints.append(training_size)
        yPointsWithPruning.append(avgWithPruning)
        yPointsWithoutPruning.append(avgWithoutPruning)

    plt.figure()
    plt.plot(xPoints, yPointsWithoutPruning)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve Without Pruning")
    plt.savefig("withoutPruning.png")

    plt.figure()
    plt.plot(xPoints, yPointsWithPruning)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve With Pruning")
    plt.savefig("withPruning.png")
    plt.show()


#Q5
def testCars():
  train = parse.parse('cars_train.data')
  test = parse.parse('cars_test.data')
  valid = parse.parse('cars_valid.data')
  cars_tree = ID3.ID3(train, "unacc")
  
  print(ID3.test(cars_tree, train))
  print(ID3.test(cars_tree, test))
  print(ID3.test(cars_tree, valid))
  
  pruned_cars_tree = ID3.prune(cars_tree, valid)
  print(ID3.test(pruned_cars_tree, test))
  
#Q6
def testRandomForest():
  forest = RandomForest("candy.data", 7)
  return forest.compare()


if __name__ == "__main__":
  #Q4
  learning_curves()
  
  #Q5
  testCars() 

  #Q6
  #generate new forest and test againt tree - do this 100x
  forest_over_tree = 0
  for _ in range(100):
    if testRandomForest():
      forest_over_tree += 1
  print(forest_over_tree/100)