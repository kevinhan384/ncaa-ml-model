from forest import RandomForest
import read_data

def testRandomForest():
    forest = RandomForest("data/all-years-thresholded.csv", 7)
    acc = forest.testForestAcc()
    print(acc)
    
# def cross_validate():
#     years = ['14-15', '15-16','16-17','17-18','18-19','20-21', '21-22', '22-23', '23-24']
    
#     n_splits = len(years)
    
#     for i in range(n_splits):
#         train_years = years[0:i] + years[i+1:]
#         test_year = years[i]
        
#         train = []
        
#         for train_year in train_years:
#             data = read_data.read_data(f"data/{train_year}-thresholded.csv")
#             train += data
            
#         test = read_data.read_data(f"data/{test_year}-thresholded.csv")
#         valid = test
#         forest = RandomForest(train, valid, test, 7)
#         acc = forest.testForestAcc()
#         print(acc)


if __name__ == "__main__":
    testRandomForest()