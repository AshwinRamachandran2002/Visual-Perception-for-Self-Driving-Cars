# Visual Perception of Self Driving Cars

Done as a part of IIT Bombay WNCC's Summer of Code program.

A implementation of Faster RCNN and MaskRCNN for object detection and instance segmenataion using the CARLA Simulator.


### Carla Vehicle on Automation of Steering Controls




https://user-images.githubusercontent.com/77834936/125677267-bc32b6d5-e2e4-4fe9-b39c-845eb43a25df.mp4


Video is run at 2.5x to visualize

### Carla environment on Mask RCNN object detection model

![detection](https://user-images.githubusercontent.com/77834936/126024259-d3ddbcf5-0a60-4dfe-83ce-54657cc52ed5.png)



## CARLA SIMULATOR

Application of models was done in the [carla-simulator](https://github.com/carla-simulator/carla). 

## Object Detection

Object detection was done through transfer learning from matterport's implementation of the Mask RCNN. 

Fine tuning ran for 6 epochs on the additional training data. 

More information from corresponding README
## Self Driving Simulation

Insipiration for the pipeline- [NVIDIA PilotNet](https://arxiv.org/pdf/1604.07316.pdf)

More information from corresponding README




