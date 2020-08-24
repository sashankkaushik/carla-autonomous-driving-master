import numpy as np



class Reward():

    def __init__(self):
        pass
    

    def get_reward(self, data, action):
        '''
        The main method we call for getting reward
        '''

        # The data we get from carla
        image, depth, seg, velocity, accleration = data
         
        # Our car position in the lane
        position = self.lane_position(image)

        # The distances of few objects (5-10) from our car
        distances = self.object_distances(image, depth)
        
        # The final reward
        reward = self.calculate_reward(position, distances, velocity) 

        # print("Inside get_reward", reward)
        # return reward

        return 1


    
    def lane_position(self, image):
        '''
        Detect lanes
        calculate our car position wrt the lanes 
        
        [Use helper functions to keep this minimal]
        Implement in lane_position.py
        
        '''
        position = 0
        print("Inside lane position", image.shape)

        return position

    
    def object_distances(self, image, depth):
        '''
        We can return the distances of the closest 5-10 objects
        
        Lets say our car is in traffic, our agent should be able to 
        apply brakes and move at low speed
        
        First find the objects, then calculate distances based on the
        depth from our camera 
        
        [Use helper functions to keep this minimal] 
        Implement in object_distances.py

        '''

        distances = [0,101,20]
        print("Inside object_distances", image.shape, depth.shape)

        return distances

    def detect_signal(self, data):
        '''
        When detected we reward our agent based on the action performed near the signal

        [Use helper functions to keep this minimal] 
        Implement in detect_signal.py

        '''

        print("Inside detect_signal", len(data))

        return bool



    def calculate_reward(self, position, distances, velocity):
        '''
        Returns the total reward

        Each reward helps the agent to learn to perfrom better actions in each case

        '''
        print("Inside calculate_reward")

        ###############################################################
        '''
        Reward for Lane position :
        Helps in lane keeping

        We penalize according to the percentage occupancy of the lane

        [Use helper functions and just get reward here using lane positon]

        Implement in lane_reward.py

        '''

        rew_pos = None

        ###############################################################
        '''
        Reward for Distances :
        Helps in maintaining minimal distance from other objects and applying brake

        We should penalize the car for coming very close to objects like pedestrians
        or other moving cars

        [Use helper functions and just get reward here from using distances]

        Implement in "distances_reward.py"

        '''

        rew_dis = None

        ###############################################################
        '''
        Reward for Speed : 
        Helps in controling our car speed
        
        1. We should penalize for high speed of the car if we are taking turn or overspeeding
        2. We should also consider the object distances to reward the speed of car (speed should be less in traffic or crowded areas)

        [Use helper functions and just get reward here using car speed and acceleration]
        
        Implement in "speed_reward.py"

        '''

        rew_vel = None

        ###############################################################
        '''
        Reward at signal :
        Helps in understanding when to cross the signal

        1. If signal detected we add this to total reward else put it as 0
        2. We should give this reward when the car is crossing the signal not just when signal is detected

        [Use helper functions and just get reward here using the state of signal and action performed]
        
        Implement in "signal_reward.py"

        '''
        
        rew_sig = None

        ###############################################################
        '''
        Reward for applying brake at high speed, brake and throttle at the same time, applying reverse gear at high speed :
        Helps in learning basic knowledge on how to control a car

        [Use helper functions and just get reward using the state of signal and action performed]

        Implement in "random_action_reward.py"
        
        '''

        rew_wrek = None

        ###############################################################



        # total_reward = rew_pos + rew_dis + rew_vel + rew_sig + rew_wrek

        # return total_reward



