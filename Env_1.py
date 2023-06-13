import matplotlib
import matplotlib.pyplot as plt
import pandas
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
import mpl_toolkits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

@torch.no_grad()
class Plane:
    def __init__(self, tag, v_x_cmd, v_y_cmd, v_z_cmd, h_cmd, x_b=torch.rand(1).item(), y_b=torch.rand(1).item(), z_b=-torch.rand(1).item(), v_x=100,
                 v_y=0, v_z=0, u=torch.rand(1).item(), v=torch.rand(1).item(), w=torch.rand(1).item(), X=0, Y=0, Z=0, T_max=49840,
                 p=torch.rand(1).item(), q=torch.rand(1).item(), r=torch.rand(1).item(), l_a=torch.rand(1).item(), m_a=torch.rand(1).item(), n_a=torch.rand(1).item(), m=15119):
        self.action_space = 4
        self.observation_space = 14
        self.tag = tag
        self.v_x_cmd = v_x_cmd
        self.v_y_cmd = v_y_cmd
        self.v_z_cmd = v_z_cmd
        self.h_cmd = h_cmd
        self.m = m
        self.x_b = x_b
        self.y_b = y_b
        self.z_b = z_b
        self.x_d = [0.0]
        self.y_d = [0.0]
        self.z_d = [0.0]
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.u = u
        self.v = v
        self.w = w
        self.X = X
        self.Y = Y
        self.Z = Z
        self.T_max = T_max
        self.p = p
        self.q = q
        self.r = r
        self.p = p
        self.l_a = l_a
        self.m_a = m_a
        self.n_a = n_a
        self.g = 9.8
        self.h_err_sum = 0
        self.v_x_err_sum = 0
        self.v_y_err_sum = 0
        self.psi_err_sum = 0
        self.t = 0
        self.done = 0
        self.q_ = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).T

        # 初始姿态四元数
        self.q_0, self.q_1, self.q_2, self.q_3 = 1, 0, 0, 0
        k = 31184 * 230414 - (4028.1 ** 2)

        # 飞行器惯量张量

        self.I = torch.tensor([[31184, 0, 4028.1], [0, 205125, 0], [4028.1, 0, 230414]], dtype=torch.float32)
        self.I_1 = torch.tensor([[31184 / k, 0, 4028.1 / k], [0, 1 / 205125, 0], [4028.1 / k, 0, 230414 / k]],
                                dtype=torch.float32)

        # 飞行器体系下角速度
        self.omega = torch.tensor([[0.0001, 0.0001, 0.0001]], dtype=torch.float32).T
        # 欧拉角
        self.phi = 0.0001
        self.theta = 0.0001
        self.psi = 0.0001

        # 迎角和侧滑角
        self.alpha = 0.0001
        self.beta = 0.0001

        # 初始状态
        self.h_err = -self.z_b - self.h_cmd
        self.v_x_err = self.v_x - self.v_x_cmd
        self.v_y_err = self.v_y - self.v_y_cmd
        self.v_z_err = self.v_z - self.v_z_cmd
        self.psi_err = self.psi - np.arctan(self.v_y_cmd / self.v_x_cmd)
        self.h_err_sum += self.h_err
        self.v_x_err_sum += self.v_x_err
        self.v_y_err_sum += self.v_y_err
        self.psi_err_sum += self.psi_err

        self.state = [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi, self.theta,
                      self.psi_err.item(), self.p, self.q,
                      self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum.item()]

    def reset(self, x_b=torch.rand(1).item(), y_b=torch.rand(1).item(), z_b=-torch.rand(1).item(), v_x=100,
                 v_y=0, v_z=0, u=torch.rand(1).item(), v=torch.rand(1).item(), w=torch.rand(1).item(), X=0, Y=0, Z=0, T_max=49840,
                 p=torch.rand(1).item(), q=torch.rand(1).item(), r=torch.rand(1).item(), l_a=torch.rand(1).item(), m_a=torch.rand(1).item(), n_a=torch.rand(1).item(), m=15119):
        self.x_b = x_b
        self.y_b = y_b
        self.z_b = z_b
        self.x_d = [0]
        self.y_d = [0]
        self.z_d = [0]
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.u = u
        self.v = v
        self.w = w
        self.X = X
        self.Y = Y
        self.Z = Z
        self.T_max = T_max
        self.p = p
        self.q = q
        self.r = r
        self.p = p
        self.l_a = l_a
        self.m_a = m_a
        self.n_a = n_a
        self.g = 9.8
        self.h_err_sum = 0
        self.v_x_err_sum = 0
        self.v_y_err_sum = 0
        self.psi_err_sum = 0
        self.t = 0
        self.done = 0
        self.q_ = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).T

        # 初始姿态四元数
        self.q_0, self.q_1, self.q_2, self.q_3 = 1, 0, 0, 0
        k = 31184 * 230414 - (4028.1 ** 2)

        # 飞行器惯量张量

        self.I = torch.tensor([[31184, 0, 4028.1], [0, 205125, 0], [4028.1, 0, 230414]], dtype=torch.float32)
        self.I_1 = torch.tensor([[31184 / k, 0, 4028.1 / k], [0, 1 / 205125, 0], [4028.1 / k, 0, 230414 / k]],
                                dtype=torch.float32)

        # 飞行器体系下角速度
        self.omega = torch.tensor([[0.0001, 0.0001, 0.0001]], dtype=torch.float32).T
        # 欧拉角
        self.phi = 0.0001
        self.theta = 0.0001
        self.psi = 0.0001

        # 迎角和侧滑角
        self.alpha = 0.0001
        self.beta = 0.0001

        # 初始状态
        self.h_err = -self.z_b - self.h_cmd
        self.v_x_err = self.v_x - self.v_x_cmd
        self.v_y_err = self.v_y - self.v_y_cmd
        self.v_z_err = self.v_z - self.v_z_cmd
        self.psi_err = self.psi - np.arctan(self.v_y_cmd / self.v_x_cmd)
        self.h_err_sum += self.h_err
        self.v_x_err_sum += self.v_x_err
        self.v_y_err_sum += self.v_y_err
        self.psi_err_sum += self.psi_err

        self.state = [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi, self.theta,
                      self.psi_err.item(), self.p, self.q,
                      self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum.item()]
        return self.state

    def step(self, action, delt_t):

        # print(action)
        # action[0] = action[0]*10.0
        # action[1] = action[1]*10.0
        # action[2] = action[2]*10.0
        # print(action)

        action[0] = np.clip(action[0], a_min=-24.0, a_max=10.5)
        action[1] = np.clip(action[1], a_min=-25.0, a_max=25.0)
        action[2] = np.clip(action[2], a_min=-30.0, a_max=30.0)
        action[3] = np.clip(action[3], a_min=0.0, a_max=1.0)

        delta_e = action[0]
        delta_a = action[1]
        delta_r = action[2]
        eta = action[3]
        self.t += 1

        # 计算迎角和侧滑角
        self.alpha = np.arctan(self.w / self.u)

        if self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2) > 1:
            self.beta = np.arcsin(0.99)
            self.done = 1
        elif self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2) < -1:
            self.beta = np.arcsin(-0.99)
            self.done = 1
        else:
            self.beta = np.arcsin(self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2))

        # 气动力模型
        if self.alpha >= -5 and self.alpha < 20:
            C_D = 0.0013 * (self.alpha ** 2) - 0.00438 * self.alpha + 0.1423
        elif self.alpha >= 20 and self.alpha <= 40:
            C_D = -0.0000348 * (self.alpha ** 2) + 0.0473 * self.alpha - 0.358
        else:
            C_D = -0.0000348 * (self.alpha ** 2) + 0.0473 * self.alpha - 0.358
            self.done = 1

        C_Y = -0.0186 * self.beta + (delta_a / 25) * (-0.00227 * self.alpha + 0.039) + (delta_r / 30) * (
                -0.00265 * self.alpha + 0.141)

        if self.alpha >= -5 and self.alpha < 10:
            C_L = 0.0751 * self.alpha + 0.0144 * delta_e + 0.732
        elif self.alpha >= 10 and self.alpha <= 40:
            C_L = -0.00148 * (self.alpha ** 2) + 0.106 * self.alpha + 0.0144 * delta_e + 0.569
        else:
            C_L = -0.00148 * (self.alpha ** 2) + 0.106 * self.alpha + 0.0144 * delta_e + 0.569
            self.done = 1

        if self.alpha >= -5 and self.alpha < 15:
            C_l = (-0.00012 * self.alpha - 0.00092) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                    0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
        elif self.alpha >= 15 and self.alpha <= 25:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                    0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
        else:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                    0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
            self.done = 1

        C_m = -0.00437 * self.alpha - 0.0196 * delta_e - 0.123 * self.q - 0.1885

        if self.alpha >= -5 and self.alpha < 10:
            C_n = 0.00125 * self.beta - 0.0142 * self.r + (delta_a / 25) * (0.000213 * self.alpha + 0.00128) + (
                    delta_r / 30) * (0.000804 * self.alpha - 0.0474)
        elif self.alpha >= 10 and self.alpha < 25:
            C_n = (-0.00022 * self.alpha + 0.00342) * self.beta - 0.0142 * self.r + (delta_a / 25) * (
                    0.000213 * self.alpha + 0.00128) + (delta_r / 30) * (0.000804 * self.alpha - 0.0474)
        elif self.alpha >= 25 and self.alpha <= 35:
            C_n = -0.00201 * self.beta - 0.0142 * self.r + (delta_a / 25) * (0.000213 * self.alpha + 0.00128) + (
                    delta_r / 30) * (0.000804 * self.alpha - 0.0474)
        else:
            C_n = -0.00201 * self.beta - 0.0142 * self.r + (delta_a / 25) * (0.000213 * self.alpha + 0.00128) + (
                    delta_r / 30) * (0.000804 * self.alpha - 0.0474)
            self.done = 1

        # print('delta_e', delta_e)
        # print('delta_a', delta_a)
        # print('delta_r', delta_r)
        # print('eta', eta)
        # print('alpha', self.alpha)
        # print('beta', self.beta)
        # print('C_D', C_D)
        # print('C_Y', C_Y)
        # print('C_L', C_L)
        # print('C_l', C_l)
        # print('C_m', C_m)
        # print('C_n', C_n)
        assert delta_a != np.nan
        assert delta_e != np.nan
        assert delta_r != np.nan
        assert eta != np.nan
        assert self.alpha != np.nan
        assert self.beta != np.nan
        assert C_D != np.nan
        assert C_Y != np.nan
        assert C_L != np.nan
        assert C_l != np.nan
        assert C_m != np.nan
        assert C_n != np.nan


        if self.done == 0:
            rho = 1.293
            V = np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)
            Q = 0.5 * rho * (V ** 2)
            S = 37.16
            b = 11.41
            c = 3.51

            self.D = Q * S * C_D
            self.Y = Q * S * C_Y
            self.L = Q * S * C_L

            self.l_a = Q * S * b * C_l
            self.m_a = Q * S * c * C_m
            self.n_a = Q * S * b * C_n

            # 更新飞行器体系下角速度

            M = torch.tensor([[self.l_a, self.m_a, self.n_a]], dtype=torch.float32).T
            pqr = torch.tensor([[self.p], [self.q], [self.r]], dtype=torch.float32)
            pqr_m = torch.tensor([[0, -self.r, self.q], [self.r, 0, -self.p], [-self.q, self.p, 0]],
                                 dtype=torch.float32)
            omega_plus = delt_t * (self.I_1 @ (-1 * pqr_m @ self.I @ pqr + M))

            self.omega = self.omega + omega_plus

            self.p = self.p + omega_plus[0].item()
            self.q = self.q + omega_plus[1].item()
            self.r = self.r + omega_plus[2].item()

            # 更新姿态四元数
            self.q_ = self.q_ + delt_t * 0.5 * (torch.tensor([[0, -self.p, -self.q, -self.r],
                                                                [self.p, 0, self.r, -self.q],
                                                                [self.q, -self.r, 0, self.p],
                                                                [self.r, self.q, -self.p, 0]], dtype=torch.float32) @ self.q_)

            self.q_0 = self.q_[0]
            self.q_1 = self.q_[1]
            self.q_2 = self.q_[2]
            self.q_3 = self.q_[3]

            # 根据四元数计算欧拉角
            # self.phi = np.arctan((2 * self.q_0 * self.q_1 + 2 * self.q_2 * self.q_3) / (
            #         self.q_0 ** 2 - self.q_1 ** 2 - self.q_2 ** 2 + self.q_3 ** 2))
            # if 2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3 <= -1:
            #     self.theta = np.arcsin(-0.99)
            # elif 2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3 >= 1:
            #     self.theta = np.arcsin(0.99)
            # else:
            #     self.theta = np.arcsin(2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3)
            # self.psi = np.arctan((2 * self.q_0 * self.q_3 + 2 * self.q_1 * self.q_2) / (
            #         self.q_0 ** 2 + self.q_1 ** 2 - self.q_2 ** 2 - self.q_3 ** 2))

            self.phi = np.arctan2(2 * (self.q_0 * self.q_1 + self.q_2 * self.q_3),1- 2 * (self.q_1 ** 2 + self.q_2 ** 2))
            input = np.clip(2 * (self.q_0 * self.q_2 - self.q_3 * self.q_1), -1, 1)
            self.theta = np.arcsin(input)
            self.psi = np.arctan2(2 * (self.q_0 * self.q_3 + self.q_1 * self.q_2),1 - 2 * (self.q_2 ** 2 + self.q_3 ** 2))

            if self.theta > 90 or self.theta < -90 or self.psi < -90 or self.psi > 90:
                self.done = True

            # 方向余弦矩阵
            # self.R_b_g = torch.tensor(
            #     [[np.cos(self.phi), 0, -np.sin(self.phi)], [0, 1, 0], [np.sin(self.phi), 0, np.cos(self.phi)]],
            #     dtype=torch.float32) @ torch.tensor(
            #     [[1, 0, 0], [0, np.cos(self.theta), np.sin(self.theta)], [0, -np.sin(self.theta), np.cos(self.theta)]],
            #     dtype=torch.float32) @ torch.tensor(
            #     [[np.cos(self.psi), np.sin(self.psi), 0], [-np.sin(self.psi), np.cos(self.psi), 0], [0, 0, 1]],
            #     dtype=torch.float32)

            self.R_b_g = torch.tensor([[np.cos(self.theta)*np.cos(self.psi), np.cos(self.theta)*np.sin(self.psi), -np.sin(self.theta)],
                                       [np.sin(self.phi)*np.sin(self.theta)*np.cos(self.psi)-np.cos(self.phi)*np.sin(self.psi), \
                                        np.sin(self.phi)*np.sin(self.theta)*np.sin(self.psi)+np.cos(self.phi)*np.cos(self.psi), np.sin(self.phi)*np.cos(self.theta)],
                                       [np.cos(self.phi)*np.sin(self.theta)*np.cos(self.psi)+np.sin(self.phi)*np.sin(self.psi), \
                                        np.cos(self.phi)*np.sin(self.theta)*np.sin(self.psi)-np.sin(self.phi)*np.cos(self.psi), np.cos(self.phi)*np.cos(self.theta)]],
                                      dtype=torch.float32)



            self.R_g_b = self.R_b_g.transpose(0, 1)

            XYZ = torch.tensor([[np.cos(self.alpha)*np.cos(self.beta) , np.sin(self.beta) , np.sin(self.alpha)*np.cos(self.beta)],
                                [-np.cos(self.alpha)*np.sin(self.beta), np.cos(self.beta), -np.sin(self.alpha)*np.sin(self.beta)],
                                [-np.sin(self.alpha), 0, np.cos(self.alpha)]], dtype=torch.float32).T @ torch.tensor([[-self.D], [self.Y], [-self.L]], dtype=torch.float32)


            self.X = XYZ[0]
            self.Y = XYZ[1]
            self.Z = XYZ[2]

            self.u = self.u + delt_t * (self.v * self.r - self.w * self.q - self.g * self.R_b_g[0][2] + (
                    self.X + eta * self.T_max) / self.m)
            self.v = self.v + delt_t * (
                    -self.u * self.r + self.w * self.p - self.g * self.R_b_g[1][2] + self.Y / self.m)
            self.w = self.w + delt_t * (self.u * self.q - self.v * self.p - self.g * self.R_b_g[2][2] + self.Z / self.m)

            uvw = torch.tensor([[self.u, self.v, self.w]], dtype=torch.float32).T
            vxyz = self.R_g_b @ uvw

            self.v_x = vxyz[0].item()
            self.v_y = vxyz[1].item()
            self.v_z = vxyz[2].item()

            self.x_b =  self.x_b + delt_t * self.v_x
            self.y_b =  self.y_b + delt_t * self.v_y
            self.z_b =  self.z_b + delt_t * self.v_z

            self.x_d.append(self.x_b)
            self.y_d.append(self.y_b)
            self.z_d.append(self.z_b)


            # ax1 = Axes3D(fig, auto_add_to_figure=False)
            # fig.add_axes(ax1)
            # ax1.plot3D(self.x_d, self.y_d, self.z_d, 'red')

            self.h_err = -self.z_b - self.h_cmd
            self.v_x_err = self.v_x - self.v_x_cmd
            self.v_y_err = self.v_y - self.v_y_cmd
            self.v_z_err = self.v_z - self.v_z_cmd
            self.psi_err = self.psi - np.arctan(self.v_y_cmd / self.v_x_cmd)
            self.h_err_sum += self.h_err
            self.v_x_err_sum += self.v_x_err
            self.v_y_err_sum += self.v_y_err
            self.psi_err_sum += self.psi_err

            self.state = [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi.item(), self.theta.item(),
                          self.psi_err.item(), self.p, self.q,
                          self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum.item()]

            # self.reward = (0.01 * np.abs(self.h_err) + 0.01 * (np.abs(self.v_x_err) + \
            #             np.abs(self.v_y_err) + np.abs(self.v_z_err)) + 2 * ((np.abs(self.psi_err) + np.abs(self.beta) + np.abs(self.phi)))\
            #                + (np.abs(self.p) +np.abs( self.q) + np.abs(self.r))+  0.05*(np.abs(self.h_err_sum) + np.abs(self.v_x_err_sum) + np.abs(self.v_y_err_sum) + np.abs(self.psi_err_sum.item())))

            self.reward = np.array([0.01 * self.h_err])

            # print(self.reward,'1')
            # print(self.t,'2')
            # print(self.x_b,'3')
            # print(self.y_b ,'4')
            # print(self.z_b ,'5')


            assert self.h_err != np.nan
            assert self.v_x_err != np.nan
            assert self.v_y_err != np.nan
            assert self.v_z_err != np.nan
            assert self.psi_err != np.nan
            assert self.h_err_sum != np.nan
            assert self.v_x_err_sum != np.nan
            assert self.v_y_err_sum != np.nan
            assert self.psi_err_sum != np.nan
            assert self.state != np.nan
            assert self.reward != np.nan
            assert self.done != np.nan
            assert self.t != np.nan
            assert self.x_b != np.nan
            assert self.y_b != np.nan
            assert self.z_b != np.nan
            assert self.u != np.nan
            assert self.v != np.nan
            assert self.w != np.nan
            assert self.p != np.nan
            assert self.q != np.nan
            assert self.r != np.nan
            assert self.phi != np.nan
            assert self.theta != np.nan
            assert self.psi != np.nan
            assert self.h_cmd != np.nan
            assert self.v_x_cmd != np.nan
            assert self.v_y_cmd != np.nan
            assert self.v_z_cmd != np.nan

            if self.t > 5000:
                self.done = 1
                self.reward = np.array([100])
            elif self.z_b > 0 or self.z_b < -200 or self.x_b > 200 or self.x_b < -200 or self.y_b < -300:
                self.done = 1
                self.reward = np.array([-100])

        return self.state, self.reward.item(), self.done

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 10))
    p = Plane(1,65,40,0,100)
    p.reset()
    for i in range(10000):
        print(i)
        s , r, d = p.step([0,0,0,1],0.1)
        if d == 1:
            print(p.x_d,p.y_d,p.z_d,sep='\n')
            break
    plt.show()