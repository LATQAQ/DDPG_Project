
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim
import numpy as np
import gym
import os
import torch
import numpy as np
import random
import Env_1
import Env_2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace
from IPython import display

#%%
class Actor(nn.Module):  #定义Actor网络
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_states, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_actions)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def change(self, x):
        y = torch.zeros(x.shape).to(device)
        if len(x.shape) == 1:
            y[0] = x[0]*17.25 - 6.75
            y[1] = x[1]*25.0
            y[2] = x[2]*30
            y[3] = torch.abs(x[3])
        else :
            y[:,0] = x[:,0]*17.25 - 6.75
            y[:,1] = x[:,1]*25.0
            y[:,2] = x[:,2]*30
            y[:,3] = torch.abs(x[:,3])
        return y

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = self.change(x)

        return x
#%%
class Critic(nn.Module):  #定义Critic网络
    def __init__(self, num_state_action, num_action_value=1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(num_state_action, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_action_value)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, state, action):
        # 按维数1拼接

        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
#%%
# a = Actor(14,4).to(torch.device('cuda:0'))
# b = Critic(18,1).to(torch.device('cuda:0'))
# state = torch.randn(256,14).to(torch.device('cuda:0'))
# actor_loss = -b(state,a(state)).mean()
#
# print(actor_loss)
# # print(a(state))
# # print(next(a.parameters()), 'actor')
# x1 = next(a.parameters()).clone()
# optimizer = optim.Adam(a.parameters(), lr=0.001)
# optimizer.zero_grad()
# actor_loss.backward()
# optimizer.step()
# x2 = next(a.parameters()).clone()
# print(torch.equal(x1.data, x2.data))
# # print(next(a.parameters()), 'actor')
#%%
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
#%%
def clip(x):
        y = torch.zeros(x.shape).to(device)
        if len(x.shape) == 1:
            y[0] = torch.clip(x[0],-24.0,10.5)
            y[1] = torch.clip(x[1],-25.0,25.0)
            y[2] = torch.clip(x[2],-30.0,30.0)
            y[3] = torch.clip(x[3], 0.0,1.0)
        else :
            y[:,0] = torch.clip(x[:,0],-24.0,10.5)
            y[:,1] = torch.clip(x[:,1],-25.0,25.0)
            y[:,2] = torch.clip(x[:,2],-30.0,30.0)
            y[:,3] = torch.clip(x[:,3], 0.0,1.0)
        return y
#%%
class TD3:
    def __init__(self, device, action_space, state_space, batch_size, gamma, tau, action_noise,policy_noise,policy_noise_clip,delay_time):
        self.device = device
        self.critic1 = Critic(action_space + state_space, 1).to(device)
        self.critic2 = Critic(action_space + state_space, 1).to(device)
        self.actor = Actor(state_space, action_space).to(device)
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0


        if os.path.exists('actor_dict'):
            self.actor.load_state_dict(torch.load('actor_dict'))

        if os.path.exists('critic1_dict'):
            self.critic1.load_state_dict(torch.load('critic1_dict'))

        if os.path.exists('critic2_dict'):
            self.critic2.load_state_dict(torch.load('critic2_dict'))

        self.target_critic1 = Critic(action_space + state_space, 1).to(device)
        self.target_critic2 = Critic(action_space + state_space, 1).to(device)
        self.target_actor = Actor(state_space, action_space).to(device)

        self.update_network_parameters(tau=1.0)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=5e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=5e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)


    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=torch.float).to(self.device)
        action = clip(action+noise)
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.target_actor(state)
        return action.cpu().numpy()

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            # print('no')
            return

        # critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=5e-3)
        # critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=5e-3)
        # actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        # 从经验回放中中随机采样一个批量的transition
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # print(state,'state',next_state,'next_state',action,'action',reward,'reward',done,'done')

        with torch.no_grad():
            next_action = self.target_actor(next_state)

            action_noise = torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),dtype=torch.float).to(self.device)
                # smooth noise
            action_noise = torch.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)

            next_action = clip(next_action + action_noise)

            q1_ = self.target_critic1(next_state, next_action)
            q2_ = self.target_critic2(next_state, next_action)
            q_ = torch.min(q1_, q2_)
            target_value = reward + (1.0 - done) * self.gamma * q_


        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        critic1_loss = nn.MSELoss()(q1, target_value.detach())
        critic2_loss = nn.MSELoss()(q2, target_value.detach())

        # graph=make_dot(critic1_loss,params=dict(self.critic1.named_parameters()),)  # 生成计算图结构表示
        # graph.render(filename='critic1_loss',view=False,format='png')  # 将源码写入文件，并对图结构进行渲染

        # print(self.critic_loss, 'critic_loss')

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # critic1_optimizer.zero_grad()
        # critic1_loss.backward()
        # critic1_optimizer.step()
        #
        # critic2_optimizer.zero_grad()
        # critic2_loss.backward()
        # critic2_optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        q1 = self.critic1(state, self.actor(state))
        actor_loss = -q1.mean()

        # print(actor_loss, 'actor_loss')

        # for param in self.actor.parameters():
        #     param.requires_grad = True
        # graph=make_dot(self.actor_loss,params=dict(self.actor.named_parameters()),)  # 生成计算图结构表示
        # graph.render(filename='actor_loss',view=False,format='png')  # 将源码写入文件，并对图结构进行渲染

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # actor_optimizer.zero_grad()
        # actor_loss.backward()
        # actor_optimizer.step()

        self.update_network_parameters()
#%%
def test(env, agent, test_eps, max_steps):
    print("Start Testing")
    rewards = []  # 记录所有回合的奖励

    for episode in range(test_eps):
        state = env.reset()
        ep_reward = 0

        fig = plt.figure(figsize=(4, 5))
        ax1 = Axes3D(fig)
        ax1.plot3D([0], [0], [0], 'red')

        x = []
        y = []
        z = []
        for step in range(max_steps):
            action = agent.predict_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state

            x.append(env.x_g)
            y.append(env.y_g)
            z.append(-env.z_g)

            ax1.plot3D(x, y, z, 'red')
            display.display(fig)
            plt.pause(0.001)
            display.clear_output(wait=True)

            if done:
                break
        # plt.pause(10)
        rewards.append(ep_reward)
        print(f"Episode：{episode + 1}/{test_eps}，Reward：{ep_reward:.2f}")
    print("Testing Complete")
    return rewards
#%%
def train(env, agent, train_eps, test_eps, max_steps):
    print("Start Training")
    rewards = []  # 记录所有回合的奖励
    great_performance = 0

    for episode in range(train_eps):
        state = env.reset()
        ep_reward = 0

        # if episode > 499 and episode % 500 == 0:
        #     draw = 1
        # else:
        #     draw = 0

        for step in range(max_steps):

            action = agent.sample_action(state)

            next_state, reward, done = env.step(action)

            ep_reward += reward

            agent.memory.push((state, action, reward, next_state, done))
            agent.update()
            state = next_state

            # display.clear_output(wait=True)

            # if draw :
            #     testres = test(env, agent, test_eps, max_steps)
            #     fig = plt.plot(testres)
            #     display.display(fig)

            if done:
                # display.clear_output(wait=True)
                break

        # print(ep_reward)
        if great_performance < ep_reward and episode > 500:
                great_performance = ep_reward
                torch.save(agent.target_critic1.state_dict(), 'critic1_dict')
                torch.save(agent.target_critic2.state_dict(), 'critic2_dict')
                torch.save(agent.target_actor.state_dict(), 'actor_dict')
                print('save')

        if (episode + 1) % 10 == 0:
            # print(next(agent.actor.parameters()), episode, 'actor')
            # print(next(agent.critic1.parameters()), episode, 'critic1')
            # x111 = next(agent.actor.parameters())
            # print(torch.equal(x111.data, x222.data))
            # display.clear_output(wait=True)

            print(f"Episode：{episode + 1}/{train_eps}，Reward：{ep_reward:.2f}")
            # display.clear_output(wait=True)
        rewards.append(ep_reward)
    print("Training Complete")
    return rewards
#%%
# env = gym.make('Pendulum-v1')
# env.action_space.shape[0], env.observation_space.shape[0]
#%%
env = Env_2.Plane(0,100,65,40,0)
#%%
# 超参数
train_eps = 6000000
test_eps = 1
max_steps = 500000
gamma = 0.99
batch_size = 512
device = torch.device('cuda:0')
tau = 0.001

#%%
agent = TD3(device, env.action_space, env.observation_space, batch_size = batch_size, gamma= gamma, tau= tau, action_noise=1, policy_noise=0.5, policy_noise_clip=1,delay_time=2)
train_res = train(env, agent, train_eps, test_eps, max_steps)
torch.save(agent.target_critic1.state_dict(), 'critic1_dict')
torch.save(agent.target_critic2.state_dict(), 'critic2_dict')
torch.save(agent.target_actor.state_dict(), 'actor_dict')