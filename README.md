# global detection for the image contrast enhancement


This is the official TensorFlow implementation of MPNet.

[See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/abs/1901.09891). 

## Instruction

This project is for the global detection for the image contrast enhancement. The input is the image and the output is the score of if part of the image has been manipulated.

We designed a multi-path network based on the VGG network for the detection. And the input for the network is the histogram of the image to remove the influence of the image content.

## Project Structure

The structure of the folder is as following：

+ MPNet	
+ model   
+ netForImage

The MPNet contains all the codes for training and testing.

 For convenience, we integrate all the four models into one code in the Intergrity_Integration.

The mimetypes-subset is the subset of the test dataset contains all types of the images in the testing dataset.

## Demon

### Running training
You can run the test code with the following:
```
    python Intergrity_Integration/test_code.py --imagePath path_of_the_image  
```
The path_of_the_image is the path of the image, defult as the test.jpg under the direction.

### Running testing

## Dependencis
+ python == 3.6.4
+ tensorflow == 1.3.0
+ rawkit == 0.6.0
+ tflearn == 0.3.2
+ cv2 == 2.4.11
+ sklearn == 0.18.1
+ scipy == 0.19.0
