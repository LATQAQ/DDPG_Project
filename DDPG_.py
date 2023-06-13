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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace
from IPython import display

# %%
# 超参数
train_eps = 2000
test_eps = 20
max_steps = 100000
gamma = 0.99
batch_size = 128
device = torch.device('cuda')
tau = 1e-3


# %%
class Actor(nn.Module):  # 定义Actor网络
    def __init__(self, num_states, num_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_states, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, num_actions)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        y = torch.zeros_like(x)
        if len(x.shape) == 1:
            y[0] = torch.clamp(x[0], -24.0, 10.5)
            y[1] = torch.clamp(x[1], -25.0, 25.0)
            y[2] = torch.clamp(x[2], -30.0, 30.0)
            y[3] = torch.clamp(x[3], 0.0, 1.0)
        else:
            y[:, 0] = torch.clamp(x[:, 0], -24.0, 10.5)
            y[:, 1] = torch.clamp(x[:, 1], -25.0, 25.0)
            y[:, 2] = torch.clamp(x[:, 2], -30.0, 30.0)
            y[:, 3] = torch.clamp(x[:, 3], 0.0, 1.0)
        return y


# %%
class Critic(nn.Module):  # 定义Critic网络
    def __init__(self, num_state_action, num_action_value=1, init_w=3e-2):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(num_state_action, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, num_action_value)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# %%
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

    def sample(self, batch_size: int, sequential: bool = True):
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


# %%
class DDPG:
    def __init__(self, device, action_space, state_space, batch_size, gamma, tau):
        self.device = device
        self.critic = Critic(action_space + state_space, 1).to(device)
        self.actor = Actor(state_space, action_space).to(device)

        if os.path.exists('actor_dic'):
            self.actor.load_state_dict(torch.load('actor_dic'))
            self.critic.load_state_dict(torch.load('critic_dic'))

        self.target_critic = Critic(action_space + state_space, 1).to(device)
        self.target_actor = Actor(state_space, action_space).to(device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-2)

        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        return action.cpu().numpy()

    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            # print('no')
            return
        # 从经验回放中中随机采样一个批量的transition
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # print(state,'state',next_state,'next_state',action,'action',reward,'reward',done,'done')

        # 计算下一时刻的预测动作价值
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())

        # 计算y_t
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        # 计算critic_loss
        # print(state, 'state', action, 'action')
        actual_value = self.critic(state, action)

        # print(actual_value, 'actual_value', expected_value, 'expected_value')

        critic_loss = nn.MSELoss()(actual_value, expected_value.detach())
        # print("critic_loss",critic_loss)

        # 计算actor_loss
        actor_loss = self.critic(state, self.actor(state))
        actor_loss = - actor_loss.mean()
        # actor_loss = torch.clamp(actor_loss, -np.inf, np.inf)
        # print('actor_loss',actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # critic_loss = torch.clamp(critic_loss, -np.inf, np.inf)

        # graph=make_dot(critic_loss,params=dict(self.critic.named_parameters()),)  # 生成计算图结构表示
        # # graph=make_dot(out,params=dict(model.named_parameters()),show_attrs=True,show_saved=True)  # 生成计算图结构表示
        # # graph=make_dot(out)  # 生成计算图结构表示
        # # graph=make_dot(out,dict(list(model.named_parameters())+[('x',inp_tensor)]))  # 生成计算图结构表示
        # graph.render(filename='critic_loss',view=False,format='png')  # 将源码写入文件，并对图结构进行渲染
        # # filename：默认生成文件名为filename+'.gv'.s
        # # view：表示是否使用默认软件打开生成的文件
        # # format：表示生成文件的格式，可为pdf、png等格式
        #
        # graph=make_dot(actor_loss,params=dict(self.actor.named_parameters()),)  # 生成计算图结构表示
        # graph.render(filename='actor_loss',view=False,format='png')  # 将源码写入文件，并对图结构进行渲染

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


# %%
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
            next_state, reward, done = env.step(action, 0.1)
            ep_reward += reward
            state = next_state

            x.append(env.x_b)
            y.append(env.y_b)
            z.append(-env.z_b)

            ax1.plot3D(x, y, z, 'red')
            display.display(fig)
            plt.pause(0.001)
            display.clear_output(wait=True)

            if done:
                break
        rewards.append(ep_reward)
        print(f"Episode：{episode + 1}/{test_eps}，Reward：{ep_reward:.2f}")
    print("Testing Complete")
    return rewards


# %%
def train(env, agent, train_eps, max_steps):
    print("Start Training")
    rewards = []  # 记录所有回合的奖励

    for episode in range(train_eps):

        state = env.reset()
        ep_reward = 0
        x = []
        y = []
        z = []

        if episode > 50:
            # fig=plt.figure(figsize=(4,5))
            # ax1 = Axes3D(fig)
            print(next(agent.actor.parameters()), episode, 'actor')
            print(next(agent.critic.parameters()), episode, 'critic')
            draw = 1
        else:
            draw = 0

        for step in range(max_steps):
            # print(state,'state')
            action = agent.sample_action(state)
            # print(action,'action')
            # print(action.shape,'action.shape')
            action = action + np.random.normal(0, 10, action.shape)

            next_state, reward, done = env.step(action, 0.1)
            # print(reward)
            ep_reward += reward
            x.append(env.x_b)
            y.append(env.y_b)
            z.append(-env.z_b)
            agent.memory.push((state, action, reward, next_state, done))
            agent.update()
            state = next_state

            # print(next_state,'next_state','episode',episode)

            # if draw:
            #     print(next(agent.actor.parameters()), episode, 'actor')
            #     print(next(agent.critic.parameters()), episode, 'critic')
            #     os.system('cls')
            #     ax1.plot3D(x,y,z, 'red')
            #     display.display(fig)
            #     plt.pause(0.001)
            #     display.clear_output(wait=True)

            if done:
                # display.clear_output(wait=True)
                break

        # print(ep_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode：{episode + 1}/{train_eps}，Reward：{ep_reward:.2f}")
        rewards.append(ep_reward)
        # print("X",x[-1])
        # print('--------------------------------------------------')
        # print("Y",y[-1])
        # print('--------------------------------------------------')
        # print("Z",z[-1])
        # print('--------------------------------------------------')
    print("Training Complete")
    return rewards


# %%
if __name__ == '__main__':
    env = Env.Plane(1, 65, 40, 0, 100)

    agent = DDPG(device, env.action_space, env.state_space, batch_size, gamma, tau)
    train_res = train(env, agent, train_eps, max_steps)
    test_res = test(env, agent, test_eps, max_steps)

    torch.save(agent.target_critic.state_dict(), 'critic_dict')
    torch.save(agent.target_actor.state_dict(), 'actor_dict')
