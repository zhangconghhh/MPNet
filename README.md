# Global Detection For The Image Contrast Enhancement


This is the official TensorFlow implementation of MPNet.

[Global Contrast Enhancement Detection via Deep Multi-Path Network](http://www.cs.albany.edu/~lsw/papers/icpr18a.pdf). 

## Instruction

This project is for the global detection for the image contrast enhancement. The input is the image and the output is the score of if part of the image has been manipulated.

We designed a multi-path network based on the VGG network for the detection. And the input for the network is the histogram of the image to remove the influence of the image content.

## Project Structure

The structure of the folder is as following：

+ MPNet: contains all the codes for training, testing and data generation.
+ model: the well trained models  
+ netForImage: code for the image input

## Demon

### Running training
For the proposed multi-stage training process, the model of the first two stage is the basic VGG model without branches. The last stage of the training is to combine the branches and conduct the multi-path model 

For the first two model you can run the following code, and set the training flag in the conv layer for the fixed parameters.
```
    python trainCode.py  --modelName='VGG_2'  --trainData='TrainData.csv' --testData='TestData.csv' --modelDir='./model/' --forFusion=False
```
In the last stage, you can run the following code to conduct the multi-path models
```
    python trainCode.py  --modelName='MPNet'  --trainData='TrainData.csv' --testData='TestData.csv' --modelDir='./model/' --forFusion=True
```
### Running testing
You can run the test code with the following:
```
    python trainCode.py  --imagePath='test.jpg'  --netName='VGG_Fusion' 
```

## Dependencis
+ python == 3.6.4
+ tensorflow == 1.3.0
+ rawkit == 0.6.0
+ tflearn == 0.3.2
+ cv2 == 2.4.11
+ sklearn == 0.18.1
+ scipy == 0.19.0
