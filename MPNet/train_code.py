from  trainSession import *
import  argparse

# The training dataset is saved in the csv file

if __name__=='__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--modelName', type=str, default='MPNet')
   parser.add_argument('--trainData', type=str, default='TrainData.csv')
   parser.add_argument('--testData', type=str, default='TestData.csv')
   parser.add_argument('--modelDir', type=str, default='./model/')
   parser.add_argument('--forFusion', type=str, default='False')
   args = parser.parse_args()

   model_name = args.modelName
   train_data = args.trainData
   test_data = args.testData
   save_path = args.modelDir
   net_name = args.netName
   flagForFintue = args.forFinetue

   if flagForFusion:
      score = trainFinetune(train_data, test_data,  model_name, save_path)
   else:
      score = trainInitial(train_data, test_data, model_name, save_path)
   print score
