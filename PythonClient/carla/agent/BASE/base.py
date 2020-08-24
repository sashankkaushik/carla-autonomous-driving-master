import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import numpy as np
import os
import sys
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join('..', '..','..','rewards')))
from reward import Reward
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent import Agent



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    '''
    Buffer to store the experiance
    '''
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]


class NeuralNet(nn.Module):
    '''
    Make the CNN here
    '''
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class BASE(Agent):
    '''
    This is our agent
    '''

    ############################## Define variables to be constant or use argparser ##############################
    
    # Initialize the model
    # Load the model's weights if exists
    # Define all the requied parameters. 
    def __init__(self, args):

        self.learning_rate = args.learning_rate
        self.betas = args.betas
        self.alpha = args.alpha
        self.gamma = args.gamma

        input_size = 784
        hidden_size = 500
        num_classes = 10
        
        self.policy = NeuralNet(input_size, hidden_size, num_classes).to(device)
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.learning_rate, betas=self.betas)

        
        ############################## Include in every agent ###################################
        # If checkpoints exist loading them and training
        if args.load_ckpt is not None:
            self.policy.load_state_dict(torch.load(PATH1))
            print(" Checkpoints exits loading them.... ")
                
        else:
            print(" Checkpoints not found, training from scratch.... ")
        #########################################################################################        
        
        

    ############################## Mostly Fixed ##############################
    def select_action(self, state):
        action = self.policy.forward(state)
        return action

    ############################## To be change according to RL algorithm ##############################
    def update(self, memory, writer, time_step):
        '''
        Here we write the code for loss, run optimizer 
        '''
        
        # calculate loss
        loss = None

        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        ############################## Include in every agent ###################################
        # Summary of the training logging in tensorboardX
        writer.add_scalar('loss', loss, time_step)
        #########################################################################################        


    ############################## Include in every agent ###################################
    # Will be called in train.py after every 'save_freq' time steps
    def save_model(self):
        torch.save(self.policy.state_dict(), './BASE_Carla.pth')
    #########################################################################################




if __name__=='__main__':
    model = NeuralNet().to(device)