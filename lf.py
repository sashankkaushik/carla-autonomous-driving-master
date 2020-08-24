import glob
import os
import sys
import random
import time
import numpy as np
import cv2

sys.path.append(glob.glob('../carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg')[0])

import carla

from carla import ColorConverter as cc


SHOW_PREVIEW = 0
IM_WIDTH = 640
IM_HEIGHT = 480



class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    actor_list = []

    front_camera = None
    front_depth_camera = None
    front_seg_camera = None
    collision_hist = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)



        transform = carla.Transform(carla.Location(x=2.5, z=1))

        self.spawn_camera(transform)

        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        
        while self.front_depth_camera is None:
            time.sleep(0.01)
        while self.front_seg_camera is None:
            time.sleep(0.01)

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=.1))

        return self.front_camera



    def spawn_camera(self,transform):
        ##########RGB CAM##########################
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        ##########SEG CAM##########################
        self.seg_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.seg_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.seg_cam.set_attribute('fov', '110')
        self.seg_sensor = self.world.spawn_actor(self.seg_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.seg_sensor)
        self.seg_sensor.listen(lambda data: self.seg_filter(data))
        ##########Depth CAM##########################
        self.depth_cam = self.world.get_blueprint_library().find('sensor.camera.depth')
        self.depth_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.depth_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.depth_cam.set_attribute('fov', '110')
        self.depth_sensor = self.world.spawn_actor(self.depth_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.depth_sensor)
        self.depth_sensor.listen(lambda data: self.process_depth_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
        self.front_camera = i3

    def process_depth_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
        normalized = (i3[:,:,2] + i3[:,:,1] * 256 + i3[:,:,0] * 256 * 256) / (256 * 256 * 256 - 1)
        self.front_depth_camera = 255*normalized

    def seg_filter(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        i3[:,:,0] = 255*(i3[:,:,2] == 6)
        i3[:,:,1] = 255*(i3[:,:,2] == 7)
        i3[:,:,2] = 255*(i3[:,:,2] == 10)
        self.front_seg_camera = i3
        print(self.front_seg_camera.shape)

    def destroy(self):
        for actor in self.actor_list:
            print(1)
            actor.destroy()






env = CarEnv()

env.reset()

print(env.front_camera.shape)
cv2.imshow("",env.front_camera)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("hello.png",env.front_camera)

while(1):
    cv2.imshow("Front Camera",env.front_camera)
    cv2.imshow("Depth Camera",env.front_depth_camera)
    cv2.imshow("Seg Camera",env.front_seg_camera)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
env.destroy()