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


class Plane:
    def __init__(self, tag, v_x_cmd, v_y_cmd, v_z_cmd, h_cmd, x_b=0.0001, y_b=0.0001, z_b=-0.0001, v_x=0.0001,
                 v_y=0.0001, v_z=0.0001, u=0.0001, v=0.0001, w=0.0001, X=0.0001, Y=0.0001, Z=0.0001, T_max=49840,
                 p=0.0001, q=0.0001, r=0.0001, l_a=0.0001, m_a=0.0001, n_a=0.0001, m=15119):
        self.action_space = 4
        self.state_space = 14
        self.tag = tag
        self.v_x_cmd = v_x_cmd
        self.v_y_cmd = v_y_cmd
        self.v_z_cmd = v_z_cmd
        self.h_cmd = h_cmd
        self.m = m
        self.x_b = x_b
        self.y_b = y_b
        self.z_b = z_b
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
        self.done = False
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

        self.state = np.array(
            [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi, self.theta, self.psi_err, self.p, self.q,
             self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum])

    def reset(self, x_b=0.0001, y_b=0.0001, z_b=-0.0001, v_x=0.0001, v_y=0.0001, v_z=0.0001, u=0.0001, v=0.0001,
              w=0.0001, X=0.0001, Y=0.0001, Z=0.0001, T_max=49840, p=0.0001, q=0.0001, r=0.0001, l_a=0.0001, m_a=0.0001,
              n_a=0.0001, m=15119):
        self.x_b = x_b
        self.y_b = y_b
        self.z_b = z_b
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
        self.done = False
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

        self.state = np.array(
            [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi, self.theta, self.psi_err, self.p, self.q,
             self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum])
        return self.state

    def step(self, action, delt_t):

        delta_e = action[0]
        delta_a = action[1]
        delta_r = action[2]
        eta = action[3]
        self.t += 1

        # 计算迎角和侧滑角
        self.alpha = np.arctan(self.w / self.u)

        if self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2) > 1 :
            self.beta = np.arcsin(0.99)
            self.done = True
        elif self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)  < -1:
            self.beta = np.arcsin(-0.99)
            self.done = True
        else:
            self.beta = np.arcsin(self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2))

        # 气动力模型
        if self.alpha >= -5 and self.alpha < 20:
            C_D = 0.0013 * (self.alpha ** 2) - 0.00438 * self.alpha + 0.1423
        elif self.alpha >= 20 and self.alpha <= 40:
            C_D = -0.0000348 * (self.alpha ** 2) + 0.0473 * self.alpha - 0.358
        else:
            self.done = True

        C_Y = -0.0186 * self.beta + (delta_a / 25) * (-0.00227 * self.alpha + 0.039) + (delta_r / 30) * (
                    -0.00265 * self.alpha + 0.141)

        if self.alpha >= -5 and self.alpha < 10:
            C_L = 0.0751 * self.alpha + 0.0144 * delta_e + 0.732
        elif self.alpha >= 10 and self.alpha <= 40:
            C_L = -0.00148 * (self.alpha ** 2) + 0.106 * self.alpha + 0.0144 * delta_e + 0.569
        else:
            C_L = -0.00148 * (self.alpha ** 2) + 0.106 * self.alpha + 0.0144 * delta_e + 0.569
            self.done = True

        if self.alpha >= -5 and self.alpha < 15:
            C_l = (-0.00012 * self.alpha - 0.00092) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                        0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
        elif self.alpha >= 15 and self.alpha <= 25:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                        0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
        else:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (delta_a / 25) * (
                        0.00121 * self.alpha - 0.0628) - (delta_r / 30) * (0.000351 * self.alpha - 0.0124)
            self.done = True

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
            self.done = True

        if self.done == False:
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
            if omega_plus.shape[0] == 1:
                omega_plus = omega_plus.squeeze(0)
            self.omega = self.omega + omega_plus

            self.p = self.p + omega_plus[0].item()
            self.q = self.p + omega_plus[1].item()
            self.r = self.p + omega_plus[2].item()

            # 更新姿态四元数
            q_1234 = torch.tensor(
                [[-self.q_1, -self.q_2, -self.q_3], [self.q_0, -self.q_3, self.q_2], [self.q_3, self.q_0, -self.q_1],
                 [-self.q_2, self.q_1, self.q_0]], dtype=torch.float32)

            if len(q_1234.shape) > 2:
                q_1234 = q_1234.squeeze(2)
            self.q_ = self.q_ + delt_t * (0.5 * q_1234 @ self.omega)

            self.q_0 = self.q_[0]
            self.q_1 = self.q_[1]
            self.q_2 = self.q_[2]
            self.q_3 = self.q_[3]

            # 根据四元数计算欧拉角
            self.phi = np.arctan((2 * self.q_0 * self.q_1 + 2 * self.q_2 * self.q_3) / (
                    self.q_0 ** 2 - self.q_1 ** 2 - self.q_2 ** 2 + self.q_3 ** 2))
            if 2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3 <= -1:
                self.theta = np.arcsin(-0.99)
            elif 2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3 >= 1:
                self.theta = np.arcsin(0.99)
            else:
                self.theta = np.arcsin(2 * self.q_0 * self.q_2 - 2 * self.q_1 * self.q_3)
            self.psi = np.arctan((2 * self.q_0 * self.q_3 + 2 * self.q_1 * self.q_2) / (
                    self.q_0 ** 2 + self.q_1 ** 2 - self.q_2 ** 2 - self.q_3 ** 2))

            if self.theta > 90 or self.theta < -90 or self.psi < -90 or self.psi > 90:
                self.done = True

            # 方向余弦矩阵
            self.R_b_g = torch.tensor(
                [[np.cos(self.phi), 0, -np.sin(self.phi)], [0, 1, 0], [np.sin(self.phi), 0, np.cos(self.phi)]],
                dtype=torch.float32) @ torch.tensor(
                [[1, 0, 0], [0, np.cos(self.theta), np.sin(self.theta)], [0, -np.sin(self.theta), np.cos(self.theta)]],
                dtype=torch.float32) @ torch.tensor(
                [[np.cos(self.psi), np.sin(self.psi), 0], [-np.sin(self.psi), np.cos(self.psi), 0], [0, 0, 1]],
                dtype=torch.float32)

            R_1 = torch.tensor(
                [[np.cos(self.psi), np.sin(self.psi), 0], [-np.sin(self.psi), np.cos(self.psi), 0], [0, 0, 1]],
                dtype=torch.float32).T
            R_2 = torch.tensor(
                [[1, 0, 0], [0, np.cos(self.theta), np.sin(self.theta)], [0, -np.sin(self.theta), np.cos(self.theta)]],
                dtype=torch.float32).T
            R_3 = torch.tensor(
                [[np.cos(self.phi), 0, -np.sin(self.phi)], [0, 1, 0], [np.sin(self.phi), 0, np.cos(self.phi)]],
                dtype=torch.float32).T

            self.R_g_b = R_1 @ R_2 @ R_3

            self.X = np.cos(self.alpha) * np.cos(self.beta) * -self.D + (
                    -np.cos(self.alpha) * np.sin(self.beta)) * self.Y + (-np.sin(self.alpha)) * -self.L
            self.Y = np.sin(self.beta) * -self.D + np.cos(self.beta) * self.Y
            self.Z = np.sin(self.alpha) * np.cos(self.beta) * -self.D + (
                    -np.sin(self.alpha) * np.sin(self.beta)) * self.Y + (np.cos(self.alpha)) * -self.L

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

            self.x_b += delt_t * self.v_x
            self.y_b += delt_t * self.v_y
            self.z_b += delt_t * self.v_z

            self.h_err = -self.z_b - self.h_cmd
            self.v_x_err = self.v_x - self.v_x_cmd
            self.v_y_err = self.v_y - self.v_y_cmd
            self.v_z_err = self.v_z - self.v_z_cmd
            self.psi_err = self.psi - np.arctan(self.v_y_cmd / self.v_x_cmd)
            self.h_err_sum += self.h_err
            self.v_x_err_sum += self.v_x_err
            self.v_y_err_sum += self.v_y_err
            self.psi_err_sum += self.psi_err

            self.state = np.array(
                [self.h_err, self.v_x_err, self.v_y_err, self.v_z_err, self.phi.item(), self.theta.item(),
                 self.psi_err.item(), self.p, self.q,
                 self.r, self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum.item()])

            self.reward = 0.01 * np.abs(self.h_err) + 0.01 * (
                    np.abs(self.v_x_err) + np.abs(self.v_y_err) + np.abs(self.v_z_err)) + 2 * (
                                  np.abs(self.phi) + np.abs(self.beta) + np.abs(self.psi_err)) + np.abs(
                self.p) + np.abs(self.q) + np.abs(self.r) + 0.05 * (
                                  np.abs(self.h_err_sum) + np.abs(self.v_x_err_sum) + np.abs(self.v_y_err_sum) + np.abs(
                              self.v_x_err_sum))
            # print(self.reward)
            # print(self.t)
            # print(self.x_b)
            # print(self.y_b)
            # print(self.z_b)
            if self.t > 5000 or self.z_b > 0 or self.x_b < 0:
                self.done = True
            else:
                self.done = False

        return self.state, self.reward.item(), self.done


