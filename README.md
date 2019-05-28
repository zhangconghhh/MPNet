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
You can run the train code with the following:
```
    python trainCode.py  --modelName='MPNet'  --trainData='TrainData.csv' --testData='TestData.csv' --modelDir='./model/' --forFusion=False
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
