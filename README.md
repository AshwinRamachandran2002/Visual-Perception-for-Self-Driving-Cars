
# Visual Perception of Self Driving Cars

Done as a part of IIT Bombay WNCC's Summer of Code program.

A implementation of Faster RCNN and MaskRCNN for object detection and instance segmenataion using the CARLA Simulator.


### Carla Vehicle on Automation of Steering Controls



https://user-images.githubusercontent.com/77834936/125677267-bc32b6d5-e2e4-4fe9-b39c-845eb43a25df.mp4


Video is run at 2.5x to visualize

### Carla environment on Mask RCNN object detection model

![detection](https://user-images.githubusercontent.com/77834936/126024259-d3ddbcf5-0a60-4dfe-83ce-54657cc52ed5.png)

![detection2](https://user-images.githubusercontent.com/77834936/126171099-610e5fd2-243e-4978-addc-f062ad2de808.jpg)

![detection3](https://user-images.githubusercontent.com/77834936/126171039-0aeed881-24f3-45d2-88bc-fe1593cfe368.jpg)

![detection4](https://user-images.githubusercontent.com/77834936/126171047-26d8608a-dca2-40e3-81ed-de395ca5385d.jpg)

![detection4](https://user-images.githubusercontent.com/77834936/126440956-4e8c1e90-d615-46d0-9aa4-d87e088c1a56.jpg)

![detection5](https://user-images.githubusercontent.com/77834936/126440981-8c37d694-a651-42a6-855a-8b3881b91a33.jpg)

![detection6](https://user-images.githubusercontent.com/77834936/126440983-401e43c4-e3c5-43bc-a965-dc0c6675623c.jpg)

![detection7](https://user-images.githubusercontent.com/77834936/126440986-c0d7f3e5-1e18-4a5a-b974-e98071fbd793.jpg)


## CARLA SIMULATOR

Application of models was done in the [carla-simulator](https://github.com/carla-simulator/carla). 

## Object Detection

Object detection was done through transfer learning from matterport's implementation of the Mask RCNN. 

Fine tuning ran for 6 epochs on the additional training data. 

More information from corresponding README
## Self Driving Simulation

Insipiration for the pipeline- [NVIDIA PilotNet](https://arxiv.org/pdf/1604.07316.pdf)

More information from corresponding README




