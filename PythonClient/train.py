####################################### For model training #######################################

from __future__ import print_function

import numpy as np
import torch

import argparse
import logging
import random
import time
import sys
import os
from PIL import Image

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


from manual_control import CarlaGame
from manual_control import Timer
from manual_control import make_carla_settings


from rewards.reward import Reward
from carla.agent.forward_agent import ForwardAgent
from carla.agent.PPO.ppo import PPO
from carla.agent.PPO.ppo import Memory


from tensorboardX import SummaryWriter

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180



def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""

    ########################################################################################################
    # TODO
    # Change this to add more cameras
    # Remove segmentation, depth if we use our own 
    # Change the no of vehicles, pedestrians, weather
    ########################################################################################################


    ###################################### Default from carla ######################################

    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=15,
        NumberOfPedestrians=30,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(2.0, 0.0, 1.4)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(2.0, 0.0, 1.4)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)
    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)
    return settings


class CarlaRun():

    def __init__(self, client, args):
        '''
        Initialize everthing required to start the simulator and randomize the start location
        Weight will be loaded from checkpoint files 

        TODO:
        1. Add the goal location
        2. Reload the environment car stops due to some reason

        Fixed required : 
        1. Exactly specifiying the input data to the CNN, Rewards 
        2. Making CNN 
        
        '''
        
        self.carla_game = CarlaGame(client, args)

        self.manual = args.manual
        self.display = args.display
        self.EPISODE_LENGTH = args.EPISODE_LENGTH
        self.time_step = 0
        self.update_timestep = args.update_timestep
        self.memory = Memory()

        self.checkpoint_file= args.checkpoints + args.agent + '.pth'
        self.writer = SummaryWriter(self.checkpoint_file)


        
        if args.agent=='ppo':
            ############## Hyperparameters ##############
            env_name = "Carla"
            render = False
            solved_reward = 200         # stop training if avg_reward > solved_reward
            log_interval = 20           # print avg reward in the interval
            max_episodes = 50000        # max training episodes
            max_timesteps = 300         # max timesteps in one episode
            n_latent_var = 64           # number of variables in hidden layer
            update_timestep = 4000      # update policy every n timesteps
            action_std = 0.6            # constant std for action distribution
            lr = 0.0025
            betas = (0.9, 0.999)
            gamma = 0.99                # discount factor
            K_epochs = 5                # update policy for K epochs
            eps_clip = 0.2              # clip parameter for PPO
            random_seed = None
            state_dim = 20
            action_dim = 5
            #############################################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            self.agent = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip, args)
        elif args.agent=='forward_agent':
            self.agent = ForwardAgent()
        


    def execute(self, episode_num):
        """Launch the PyGame."""
        pygame.init()
        self.carla_game._initialize_game()
        try:
           
            # We should change this to max_steps in an episode 
            # It should then quit even if it doesn't reach the goal
            self.episode_reward = 0
            for i in range(self.EPISODE_LENGTH):
            # while True:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                if self.display:
                    self.carla_game._on_render()
            print("Episode reward = \t {}".format(self.episode_reward))
            writer.add_scalar('Episode Reward', self.episode_reward, episode_num)

        finally:
            pygame.quit()
        
    

    def _on_loop(self):

        ################################################################################################
        # TODO 
        # Add the required no of cameras
        # Disable prints when we switch off render,
        # 
        # if the prints are on pygame window


        ###################################### Default from carla ######################################
        self.carla_game._timer.tick()

        # Reading the data from sensors
        measurements, sensor_data = self.carla_game.client.read_data()

        self.carla_game._main_image = sensor_data.get('CameraRGB', None)
        # To get image into an array
        # if self.carla_game._main_image is not None:
            # array = image_converter.to_rgb_array(self.carla_game._main_image)
        self.carla_game._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self.carla_game._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
        self.carla_game._lidar_measurement = sensor_data.get('Lidar32', None)
        
        ###################################### Default from carla ######################################
        # TODO 
        # Check if these prints are working or not

        # Print measurements every second.
        if self.carla_game._timer.elapsed_seconds_since_lap() > 1.0:
            if self.carla_game._city_name is not None:
                # Function to get car position on map.
                map_position = self.carla_game._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self.carla_game._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self.carla_game._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self.carla_game._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.
            self.carla_game._timer.lap()

        
        ######################################## Manual control #########################################
        if self.manual:
            control = self.carla_game._get_keyboard_control(pygame.key.get_pressed())


        ######################################## Agent control ###########################################
        if not self.manual:
            # data for input
            data = [self.carla_game._main_image, self.carla_game._mini_view_image1, self.carla_game._mini_view_image2, self.carla_game._lidar_measurement]
            print("first data recieved")
            control, action = self.perform_action(data)


        ###################################### Default from carla ######################################
        # Set the player position
        if self.carla_game._city_name is not None:
            self.carla_game._position = self.carla_game._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self.carla_game._agent_positions = measurements.non_player_agents

        # passing control
        if control is None:
            self.carla_game._on_new_episode()
        elif self.carla_game._enable_autopilot:
            self.carla_game.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.carla_game.client.send_control(control)


        ######################## Get the next state readings and call run_agent ########################        
        # TODO 
        # Verify if this the next state data or not
        measurements, next_sensor_data = self.carla_game.client.read_data()

        if not self.manual:
            # we are calling this here because the state is updated 
            # after control is sent by the above commands
            print("ready to execute run_agent...")
            run_agent(data, action, next_sensor_data)



    def run_agent(self, data, action, next_sensor_data):

        # The data of the state obtained from performing action
        _main_image = next_sensor_data.get('CameraRGB', None)
        _mini_view_image1 = next_sensor_data.get('CameraDepth', None)
        _mini_view_image2 = next_sensor_data.get('CameraSemSeg', None)
        _lidar_measurement = next_sensor_data.get('Lidar32', None)
        
        # The next_state data 
        next_data = [_main_image, _mini_view_image1, _mini_view_image2, _lidar_measurement ] 
        print("Acquired next data")

        # calculating reward based on current state, action performed and next_state
        reward = self.calculate_reward(data, action, next_data)
        print("reward obtained")

        # Updating the buffer
        self.memory.rewards.append(reward)
        self.memory.states.append(data)
        self.memory.actions.append(action)

        # increasing the time step
        self.time_step += 1
        
        # update if its time
        if self.time_step % self.update_timestep == 0:
            self.agent.update(memory, self.writer, self.time_step)
            memory.clear_memory()
        
        # Reward within an episode
        self.episode_reward += reward
        

    # Used for calling the save_model in the agent class
    # This saves the weights of model
    def save(self):
        self.agent.save_model()


    def perform_action(self, data):
        '''
        Performs action

        Input : Data = [image, segmentation, depth, ...]
        Return : control
        action = [throttle, steering_angle, steering_direction, brake, reverse_gear]

        '''
        action = self.agent.select_action(data, self.memory)
        print("obtained action")

        #########################################
        # TODO
        # Add randomness in run_step of agent
        #########################################


        control = VehicleControl()

        # Car Throttle 
        control.throttle = action[0]

        ################################################################
        # TODO
        # We need to convert angle into the units which carla uses 
        # steering direction will be a bool 0 for left and 1 for right
        if action[2]:
            control.steer = function(action[1])
        else:
            control.steer = -1*function(action[1])
        ################################################################

        # brake is a bool
        control.brake = action[3]

        # If needed there is an option for hand_brake as well
        # control.hand_brake

        # Car reverse gear
        if action[4]:
            self.carla_game._is_on_reverse = not self.carla_game._is_on_reverse   
        control.reverse = self.carla_game._is_on_reverse

        print("returning control..")
        return control, action

    
    def calculate_reward(self, state, action, next_state):
        '''
        Calculate rewards
        Using the current state, the performed action and the obtained new state we calculate rewards.

        Input : current_state, action, next_state
        Return : reward (a value)

        '''
        rew = Reward()
        reward = rew.get_reward([state, action, next_state])
        return reward


  

def main():


    argparser = argparse.ArgumentParser(description='CARLA Training framework ')

    argparser.add_argument( '--render', default=False, help='Option for enabling or disabling the carla window')
    argparser.add_argument( '--host', metavar='H', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument( '-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument( '-q', '--quality-level', choices=['Low', 'Epic'], type=lambda s: s.title(), default='Epic', help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument( '-l', '--lidar', action='store_true', default=True, help='enable Lidar')
    argparser.add_argument( '-m', '--map', action='store_true',  default=False, help='plot the map of the current city')
    argparser.add_argument( '--agent', type=str, default='ppo', help='Specify the agent')
    argparser.add_argument( '-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument( '--manual', default=False, help='to enable or diable manual mode')
    argparser.add_argument( '--display', type = bool, default=False, help='to enable or diable display')
    argparser.add_argument( '--EPISODE_LENGTH', type = int, default=5000, help='to enable or diable display')
    argparser.add_argument( '--update_timestep', type = int, default=4000, help='time steps after which network is updated')
    argparser.add_argument( '--save_freq', type = int, default=5000, help='time steps after which network weights are saved')
    argparser.add_argument( '--checkpoints', type=str, default='./checkpoints', help='path to checkpoints folder')



    args = argparser.parse_args()



    '''
    We should optimize the code further to speedup the simulation 
    '''

    episode_num = 0
    while True:

        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaRun(client, args)
                game.execute(episode_num)
                episode_num+=1

                if episode_num % args.save_freq == 0:
                    game.save() 


        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


        

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


 




    ########################################################################################################
    # avg_length += t
    # # # stop training if avg_reward > solved_reward
    # if running_reward > (log_interval*solved_reward):
    #     print("########## Solved! ##########")
    #     torch.save(ppo.policy.state_dict(), './PPO_Continuous_{}.pth'.format(env_name))
    #     break
    # # logging
    # if i_episode % log_interval == 0:
    #     avg_length = int(avg_length/log_interval)
    #     running_reward = int((running_reward/log_interval
    #     print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
    #     running_reward = 0
    #     avg_length = 0
    
    # ########################################################################################################