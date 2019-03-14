import argparse
from trainFunc import *


if __name__=='__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--modelName', type=str, default='netForImage')
   parser.add_argument('--trainPath', type=str, default='./Dataset/train')
   parser.add_argument('--testPath', type=str, default='./Dataset/test')
   parser.add_argument('--modelDir', type=str, default='./model/')
   args = parser.parse_args()

   model_name = args.modelName
   train_file = args.trainData
   test_file = args.testData
   save_path = args.modelDir
   net_name = args.netName

   model_train(train_file, test_file, model_name, save_path)