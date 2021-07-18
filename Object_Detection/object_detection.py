#!/usr/bin/env python

# My implementation of Carla's object detection
# Author- Ashwin Ramachandran



"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import random
import math
import numpy as np
import cv2
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')



VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 110

ROOT_DIR = os.path.abspath("../")



# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

from mrcnn import visualize
from mrcnn.config import Config

BB_COLOR = (248, 64, 24)



def loadmodel():
    """
    Function to load pre trained model fine tuned on carla dataset
    """
    
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class CocoConfig(Config):
        """Configuration for inference on MS COCO model.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        NAME = "coco"

        IMAGES_PER_GPU = 2

        NUM_CLASSES = 1 + 1 # 1+80 for coco

    class InferenceConfig(CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        

    config = InferenceConfig()
        # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=config,
                            model_dir=ROOT_DIR)

    model_path = os.path.join(ROOT_DIR, "examples/mask_rcnn.h5")

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


model=loadmodel()


# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def draw_bounding_boxes(display, boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        N = boxes.shape[0]
        
        for i in range(N):
            print('ll')
            y1, x1, y2, x2 = boxes[i]

            
            pygame.draw.line(bb_surface, BB_COLOR,  (x1,y1),(x1,y2))
            pygame.draw.line(bb_surface, BB_COLOR,  (x1,y2),(x2,y2))
            pygame.draw.line(bb_surface, BB_COLOR,  (x2,y2),(x2,y1))
            pygame.draw.line(bb_surface, BB_COLOR,  (x2,y1),(x1,y1))
        
        display.blit(bb_surface, (0, 0))


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ============================================================================

# class-names belonging to COCO if labels are needed for display
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        
        self.display = None
        self.image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False


    @staticmethod
    def get_image_array(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((640, 480, 4))
        i3 = i2[:, :, :3]
        return i3


    def predict(self,image):

        results = model.detect([image], verbose=0)
        r = results[0]
        # image= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                             class_names, r['scores'])
        print(r['rois'])
        return r['rois']


    # @staticmethod
    def draw_boxes(self,display, boxes):
        """
        Draws bounding boxes on pygame display.
        """



        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))


        index=0
        N = boxes.shape[0]
        pygame.draw.line(bb_surface, BB_COLOR,  (10,10),(80,80))
        
        for i in range(N):
            print('ll')
            y1, x1, y2, x2 = boxes[i]

            
            pygame.draw.line(bb_surface, BB_COLOR,  (x1,y1),(x1,y2))
            pygame.draw.line(bb_surface, BB_COLOR,  (x1,y2),(x2,y2))
            pygame.draw.line(bb_surface, BB_COLOR,  (x2,y2),(x1,y2))
            pygame.draw.line(bb_surface, BB_COLOR,  (x1,y2),(x1,y1))
            

        display.blit(bb_surface, (0, 0))


    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        


        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                
                # get current image from camera on car
                imgarr=self.get_image_array(self.image)

                # predict bounding box coordinates from model
                i=self.predict(imgarr)

                #draw boxes according to result
                ClientSideBoundingBoxes.draw_bounding_boxes(self.display, i)

                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
