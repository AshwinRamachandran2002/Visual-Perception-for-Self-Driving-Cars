# Visual Perception of SelfDriving Cars

A implementation of Faster RCNN and MaskRCNN for object detection and instance segmenataion using the CARLA Simulator.



https://user-images.githubusercontent.com/77834936/125677267-bc32b6d5-e2e4-4fe9-b39c-845eb43a25df.mp4

Video is run at 2.5x to visualize

## CARLA SIMULATOR

Application of models was done in the [carla-simulator](https://github.com/carla-simulator/carla). 

## Object Detection

Object detection was done through transfer learning from matterport's implementation of the Mask RCNN. 
Training data was collected form [awesome-carla.org](https://github.com/Amin-Tgz/awesome-CARLA)

Matterport's MRCNN was changed to make it compatible with tensorflow>=2.1.0 and python>=3.5.

Fine tuning ran for 6 epochs on the additional training data. 
[Model used for training and preprocessing](object)



## Self Driving Simulation


