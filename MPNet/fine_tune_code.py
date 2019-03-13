from  train_finetune1 import *

model_name = 'vgg_dp'

# the 1 d input
train_data = '/home/congzhang/forensic_data/MFC_Dev2/dev1_3000.csv'
train_data = '/home/congzhang/forensic_data/MFC_Dev2/manipulation_dev1.csv'
train_data = '/home/congzhang/forensic_data/MFC_Dev2/data_3000.csv'
test_data = '/home/congzhang/forensic_data/MFC_Dev2/manipulation_dev2_valid.csv'
save_path = './model/'



if __name__=='__main__':

   score = model_train(train_data, test_data,  model_name, save_path)
   print score
