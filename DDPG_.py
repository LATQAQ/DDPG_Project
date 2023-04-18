
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
import Env

# 超参数
train_eps = 6000
test_eps = 20
max_steps = 5001
gamma = 0.99
batch_size = 128
device = torch.device('cuda')
tau = 1e-3

class Actor(nn.Module): #定义Actor网络
    def __init__(self, num_states, num_actions):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_states, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Critic(nn.Module): #定义Critic网络
    def __init__(self, num_state_action, num_action_value = 1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(num_state_action, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, num_action_value)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
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
class DDPG:
    def __init__(self, device, action_space, state_space, batch_size, gamma, tau):
        self.device = device
        self.critic = Critic(action_space+state_space,1).to(device)
        self.actor = Actor(state_space,action_space).to(device)
        self.target_critic = Critic(action_space+state_space,1).to(device)
        self.target_actor = Actor(state_space,action_space).to(device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity= 100000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action[0] = np.clip(action[0],a_min=-24,a_max=10.5)
        action[1] = np.clip(action[1],a_min=-25,a_max=25)
        action[2] = np.clip(action[2],a_min=-30,a_max=30)
        action[3] = np.clip(action[3],a_min=0,a_max=1)
        return action.cpu().numpy()

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中中随机采样一个批量的transition
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量

        state= torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # 计算actor_loss
        actor_loss = self.critic(state, self.actor(state))
        actor_loss = - actor_loss.mean()

        # print(actor_loss,actor_loss.shape)

        # 计算下一时刻的预测动作价值
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action)

        # 计算y_t
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        # 计算critic_loss
        actual_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(actual_value, expected_value)

        # print(critic_loss, critic_loss.shape)

        # 反向传播
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )

def test(env, agent, test_eps, max_steps):
    print("Start Testing")
    rewards = [] # 记录所有回合的奖励
    for episode in range(test_eps):
        state = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            action = agent.predict_action(state)
            next_state, reward, done = env.step(action,0.1)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode：{episode+1}/{test_eps}，Reward：{ep_reward:.2f}")
    print("Testing Complete")
    return rewards

def train(env, agent, train_eps, max_steps):
    print("Start Training")
    rewards = [] # 记录所有回合的奖励
    for episode in range(train_eps):
        state = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            action = agent.sample_action(state)
            action[0] = np.clip(action[0],a_min=-24,a_max=10.5)
            action[1] = np.clip(action[1],a_min=-25,a_max=25)
            action[2] = np.clip(action[2],a_min=-30,a_max=30)
            action[3] = np.clip(action[3],a_min=0,a_max=1)
            # print(action)
            next_state, reward, done = env.step(action,0.1)
            ep_reward += reward
            agent.memory.push((state, action, reward, next_state, done))
            agent.update()
            state = next_state
            if done:
                break

        if (episode+1)%10 == 0:
            print(f"Episode：{episode+1}/{train_eps}，Reward：{ep_reward:.2f}")
        rewards.append(ep_reward)
    print("Training Complete")
    return rewards

def draw(rewards,tag):
    sns.set(style='whitegrid')
    fig = sns.relplot(y= rewards, kind= 'line', tag=tag)
    plt.legend()

env = Env.Plane(1,65,40,0,100)

agent = DDPG(device, env.action_space, env.state_space, batch_size, gamma, tau)
train_res = train(env,agent,train_eps,max_steps)
test_res = test(env,agent,test_eps,max_steps)



# 画出结果
draw(train_res,tag="train")
draw(test_res,tag="test")