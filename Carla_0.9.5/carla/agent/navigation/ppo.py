
from agents.navigation.agent import Agent, AgentState
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gym
import argparse
import numpy as np
import os
import cv2

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/exp-1')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, n_var),
                nn.Tanh(),
                nn.Linear(n_var, n_var),
                nn.Tanh(),
                nn.Linear(n_var, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, n_var),
                nn.Tanh(),
                nn.Linear(n_var, n_var),
                nn.Tanh(),
                nn.Linear(n_var, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(torch.squeeze(action_mean), torch.diag(self.action_var))
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 15, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 1, 3)
        self.fc1 = nn.Linear(176*316, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        a,b,c,d,e = x
        print(a.shape, b.shape, c.shape, d.shape, e.shape)

        a = cv2.resize(a, dsize=(320, 180), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, dsize=(320, 180), interpolation=cv2.INTER_CUBIC)
        c = cv2.resize(c, dsize=(320, 180), interpolation=cv2.INTER_CUBIC)
        d = cv2.resize(d, dsize=(320, 180), interpolation=cv2.INTER_CUBIC)
        e = cv2.resize(e, dsize=(320, 180), interpolation=cv2.INTER_CUBIC)


        a = np.moveaxis(a,-1, 0)
        b = np.moveaxis(b,-1, 0)
        c = np.moveaxis(c,-1, 0)
        d = np.moveaxis(d,-1, 0)
        e = np.moveaxis(e,-1, 0)

        a = torch.from_numpy(a.copy())
        b = torch.from_numpy(b.copy())
        c = torch.from_numpy(c.copy())
        d = torch.from_numpy(d.copy())
        e = torch.from_numpy(e.copy())

        a = a.type('torch.FloatTensor')
        b = b.type('torch.FloatTensor')
        c = c.type('torch.FloatTensor')
        d = d.type('torch.FloatTensor')
        e = e.type('torch.FloatTensor')


        x = torch.cat([a,b],0)
        x = torch.cat([x,c],0)
        x = torch.cat([x,d],0)
        x = torch.cat([x,e],0)
        x = x.unsqueeze(0)
        x = x.to(device)


        out = F.relu(self.conv1(x))
        print(out.shape)
        out = F.relu(self.conv2(out))
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out



class PPO(Agent):
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip, args):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(),
        #                                       lr=lr, betas=betas)
        # self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)

        self.policy = Temp().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        ############################## Include in every agent ###################################
        # If checkpoints exist loading them and training
        # define the path for checkpoints file
        self.checkpoint_file = '/home/saivinay/Documents/CarlaSimulator/PythonClient/examples/checkpoints/PPO_Carla.pth' 

        if os.path.exists(self.checkpoint_file):
            self.policy.load_state_dict(torch.load(self.checkpoint_file))
            # self.policy_old.load_state_dict(self.policy.state_dict())
            print(" Checkpoints exits loading them.... ")
                
        else:
            print(" Checkpoints not found, training from scratch.... ")
        #########################################################################################

        self.MseLoss = nn.MSELoss()
    

    def select_action(self, state, memory):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        action = self.policy.forward(state).cpu().data.numpy().flatten()
        return action


    
    def update(self, memory, writer, time_step):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
     
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        ############################## Include in every agent ###################################
        # Summary of the training logging in tensorboardX
        writer.add_scalar('loss', loss, time_step)
        writer.add_scalar()
        #########################################################################################   
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


    ############################## Include in every agent ###################################
    # Will be called in train.py after every 'save_freq' time steps
    def save_model(self):
        torch.save(self.policy.state_dict(), self.checkpoint_file)
    #########################################################################################

        
def main(args):
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
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip, args)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if done:
                break
        
        avg_length += t
        
        # #########################################################################################
        # # Include in every agent
        # # Save once every save_freq
        # if i_episode % save_freq == 0:
        #     torch.save(ppo.policy.state_dict(), './PPO_Continuous_{}.pth'.format(env_name))
        # #########################################################################################


        # # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_Continuous_{}.pth'.format(env_name))
            break
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':

    #############################################################################################################
    # Include in every agent
    argparser = argparse.ArgumentParser(description='CARLA Training framework ')
    # argparser.add_argument( '--checkpoints' , type = str, default='../../../checkpoints/ppo.pth', help="path to checkpoints folder")
    args = argparser.parse_args()
    #############################################################################################################


    main(args)