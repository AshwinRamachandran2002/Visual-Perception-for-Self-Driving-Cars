# Object Detection using Mask_Rcnn


## Result of object detection
![detection](https://user-images.githubusercontent.com/77834936/126054308-a86af3ad-6cab-403a-8192-a8a93daa90cc.png)



## 1. Dataset Collection

Dataset for transfer learning was used from [ aewsome-carla ](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset)
The repository includes 800 images and corresponding xml files containing bounding box coordinates and labels.

## 2. Understanding Faster RCNN and Mask RCNN

Object Detection is to be done by using transfer learning from model trainded on COCO dataset. 

The original papers were used to understand [Mask RCNN](https://arxiv.org/abs/1703.06870) and [Faster RCNN](https://arxiv.org/abs/1506.01497)

Matterports MRCNN implementation was chosen to get inspiration for training and inference. 
Modifications were made to various modules of the original repo to make it compatible with tensorflow>=2.1.0

The modified mrcnn model can be found [here](/mrcnn)

## 3. Preprocessing and training 

Preprocessing and training was done [here](/Object_Detection.ipynb)

Training was done on two models-
1. Pretrained model trained on COCO
2. RESNET50 with Imagenet weights

The former resulted in more accuracy than latter.



Here are the [weights](https://drive.google.com/file/d/10Xk5-3wapfE6O2YaKTz3psicrTQau4e2/view?usp=sharing) after additional training of COCO on my dataset

## 4. Building Pipeline

Carla-Simulator was setup on Python3.5 and tensorflow 2.1.0.

Client side python file was made for manual control of car and displaying results of object detection 
from image captured by camera on the car.

Resources used -
1. [Building a simple manually controlled car on carla](https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/)

2. [Official Carla Tutorial](https://carla.readthedocs.io/en/latest/start_quickstart/)

3. [Modules from awesome-carla](https://github.com/Amin-Tgz/awesome-CARLA)

4. Official client side script examples -[Control of car](/manual_control.py), [Drawing boxes on game module](client_bounding_boxes.py)




