import numpy as np
import torch


class Plane:
    def __init__(self, tag, h_cmd, v_x_cmd, v_y_cmd, v_z_cmd):
        self.tag = tag
        self.h_cmd = h_cmd
        self.v_x_cmd = v_x_cmd
        self.v_y_cmd = v_y_cmd
        self.v_z_cmd = v_z_cmd
        self.action_space = 4
        self.observation_space = 14
        self.t = 0
        self.x_g = 0
        self.y_g = 0
        self.z_g = 0
        self.v_x = 100.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.u = 100.0
        self.v = 0.0
        self.w = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.p = 0.0
        self.q = 0.0
        self.r = 0.0
        self.m = 15119.0
        self.Tmax = 49840.0
        self.q_ = np.array([[1.0], [0.0], [0.0], [0.0]])

        self.h_err_sum = 0.0
        self.v_x_err_sum = 0.0
        self.v_y_err_sum = 0.0
        self.psi_err_sum = 0.0

    def reset(self):
        self.t = 0
        self.x_g = 0
        self.y_g = 0
        self.z_g = 0
        self.v_x = 100.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.u = 100.0
        self.v = 0.0
        self.w = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.p = 0.0
        self.q = 0.0
        self.r = 0.0
        self.m = 15119.0
        self.Tmax = 49840.0
        self.q_ = np.array([[1.0], [0.0], [0.0], [0.0]])

        self.h_err_sum = 0.0
        self.v_x_err_sum = 0.0
        self.v_y_err_sum = 0.0
        self.psi_err_sum = 0.0

        h_err = -self.z_g - self.h_cmd
        v_x_err = self.v_x - self.v_x_cmd
        v_y_err = self.v_y - self.v_y_cmd
        v_z_err = self.v_z - self.v_z_cmd
        psi_err = self.psi - np.arctan2(self.v_y_cmd, self.v_x_cmd)

        state = [h_err, v_x_err, v_y_err, v_z_err, self.phi, self.theta, psi_err, self.p, self.q, self.r,
                 self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum]

        return state

    def step(self, action):
        de = action[0]  # elevator
        da = action[1]  # aileron
        dr = action[2]  # rudder
        eta = action[3]  # throttle
        dt = 0.1  # time step

        self.t += 1
        S = 37.16
        b = 11.41
        c = 3.51
        rho = 1.293
        g = 9.8
        V = np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)
        Q = 0.5 * rho * V ** 2
        I_x = 3.1184e4
        I_y = 2.05125e5
        I_z = 2.30414e5
        I_xz = -4.0281e3
        I = np.array([[I_x, 0, -I_xz], [0, I_y, 0], [-I_xz, 0, I_z]])
        omega = np.array([[self.p], [self.q], [self.r]])
        q0 = self.q_[0].item()
        q1 = self.q_[1].item()
        q2 = self.q_[2].item()
        q3 = self.q_[3].item()
        uvw = np.array([[self.u], [self.v], [self.w]])

        reward = 0.0
        done = 0.0

        if -5 <= self.alpha <= 20:
            C_D = 0.0013 * self.alpha ** 2 - 0.00438 * self.alpha + 0.1423
        elif 20 < self.alpha <= 40:
            C_D = -0.0000348 * self.alpha + 0.0473 * self.alpha - 0.358
        else:
            C_D = -0.0000348 * self.alpha + 0.0473 * self.alpha - 0.358
            done = 1

        if -5 <= self.alpha <= 10:
            C_L = 0.0751 * self.alpha + 0.0144 * de + 0.732
        elif 10 < self.alpha <= 40:
            C_L = -0.00148 * self.alpha ** 2 + 0.106 * self.alpha + 0.0144 * de + 0.569
        else:
            C_L = -0.00148 * self.alpha ** 2 + 0.106 * self.alpha + 0.0144 * de + 0.569
            done = 1

        C_Y = -0.0186 * self.beta + (da / 25.0) * (-0.00227 * self.alpha + 0.039) + (dr / 30.0) * (
                -0.00265 * self.alpha + 0.141)

        if -5 <= self.alpha < 15:
            C_l = (-0.00012 * self.alpha - 0.00092) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (da / 25.0) * (
                    0.00121 * self.alpha - 0.0628) - (dr / 30.0) * (0.000351 * self.alpha - 0.0124)
        elif 15 <= self.alpha <= 25:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (da / 25.0) * (
                    0.00121 * self.alpha - 0.0628) - (dr / 30.0) * (0.000351 * self.alpha - 0.0124)
        else:
            C_l = (0.00022 * self.alpha - 0.006) * self.beta - 0.0315 * self.p + 0.0126 * self.r + (da / 25.0) * (
                    0.00121 * self.alpha - 0.0628) - (dr / 30.0) * (0.000351 * self.alpha - 0.0124)
            done = 1

        C_m = -0.00437 * self.alpha - 0.0196 * de - 0.123 * self.q - 0.1885

        if -5 <= self.alpha < 10:
            C_n = 0.00125 * self.beta - 0.0142 * self.r + (da / 25) * (0.000213 * self.alpha + 0.00128) + (
                    dr / 30) * (0.000804 * self.alpha - 0.0474)
        elif 10 <= self.alpha < 25:
            C_n = (-0.00022 * self.alpha + 0.00342) * self.beta - 0.0142 * self.r + (da / 25) * (
                    0.000213 * self.alpha + 0.00128) + (
                          dr / 30) * (0.000804 * self.alpha - 0.0474)
        elif 25 <= self.alpha <= 35:
            C_n = -0.00201 * self.beta - 0.0142 * self.r + (da / 25) * (0.000213 * self.alpha + 0.00128) + (
                    dr / 30) * (0.000804 * self.alpha - 0.0474)
        else:
            C_n = -0.00201 * self.beta - 0.0142 * self.r + (da / 25) * (0.000213 * self.alpha + 0.00128) + (
                    dr / 30) * (0.000804 * self.alpha - 0.0474)
            done = 1

        # C_D = (1.4610*(self.alpha**4)+(-5.7341)*(self.alpha**3)+6.3971*(self.alpha**2)+(-0.1995)*(self.alpha)+(-1.4994))*np.cos(self.beta)+1.5036+(0.7771*self.alpha-0.0276)*de
        # C_L = (1.1645*self.alpha**3 -5.4246*self.alpha**2 +5.6770*self.alpha -0.0204)*np.cos(2*self.beta/3)*(-0.3573*self.alpha+0.8564)*de
        # C_Y = (0.5781*self.alpha**2 + 0.2834 * self.alpha -0.8615)*self.beta + (0.4270*self.alpha-0.1047)*da + (-0.4486*self.alpha+0.3079)*dr
        # C_l = (0.8102*self.alpha**2 -0.6446*self.alpha -0.0427)*self.beta + (-0.1553*self.alpha + 0.1542)*da + (-0.858*self.alpha+0.0943)*dr + (b/(2*V*3.2808399))*(0.0201*self.alpha -0.3370)*self.r
        # C_m = -0.9931*self.alpha + 0.1407 + (0.6401*self.alpha-1.1055)*de + (c/(2*V*3.2808399))*(-14.30*self.alpha-2.00)*self.q
        # C_n = (-0.3917*self.alpha**2 + 0.3648*self.alpha + 0.0894)*self.beta + (-0.0213*self.alpha + 0.0051)*da + (0.0534*self.alpha -0.0724)*dr + (-0.0716*self.alpha-0.4375)*(b/(2*V*3.2808399))*self.r

        D = Q * S * C_D
        Y = Q * S * C_Y
        L = Q * S * C_L
        l_a = Q * S * b * C_l
        m_a = Q * S * c * C_m
        n_a = Q * S * b * C_n

        XYZ = np.array(
            [[np.cos(self.alpha) * np.cos(self.beta), np.sin(self.beta), np.sin(self.alpha) * np.cos(self.beta)],
             [-np.cos(self.alpha) * np.sin(self.beta), np.cos(self.beta), -np.sin(self.alpha) * np.sin(self.beta)],
             [-np.sin(self.alpha), 0, np.cos(self.alpha)]]).T @ np.array([[-D], [Y], [-L]])

        X = XYZ[0].item()
        Y = XYZ[1].item()
        Z = XYZ[2].item()

        omega_dot = np.linalg.inv(I) @ (-np.array(
            [[0, -self.r, self.q], [self.r, 0, -self.p], [-self.q, self.p, 0]]) @ I @ omega + np.array(
            [[l_a], [m_a], [n_a]]))

        omega = omega + omega_dot * dt
        self.p = omega[0].item()
        self.q = omega[1].item()
        self.r = omega[2].item()

        q__dot = 0.5 * (np.array([[-q1, -q2, -q3], [q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]]) @ omega)
        self.q_ = self.q_ + q__dot * dt
        q0 = self.q_[0].item()
        q1 = self.q_[1].item()
        q2 = self.q_[2].item()
        q3 = self.q_[3].item()

        self.phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        if 2 * (q0 * q2 - q3 * q1) > 1:
            # reward += -10
            self.theta = np.pi / 2
            done = 1.0
        elif 2 * (q0 * q2 - q3 * q1) < -1:
            self.theta = -np.pi / 2
            # reward += -10
            done = 1.0
        else:
            self.theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
        self.psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

        R_b_g = np.array(
            [[np.cos(self.theta) * np.cos(self.psi), np.cos(self.theta) * np.sin(self.psi), -np.sin(self.theta)],
             [np.sin(self.phi) * np.sin(self.theta) * np.cos(self.psi) - np.cos(self.phi) * np.sin(self.psi),
              np.sin(self.phi) * np.sin(self.theta) * np.sin(self.psi) + np.cos(self.phi) * np.cos(self.psi),
              np.sin(self.phi) * np.cos(self.theta)],
             [np.cos(self.phi) * np.sin(self.theta) * np.cos(self.psi) + np.sin(self.phi) * np.sin(self.psi),
              np.cos(self.phi) * np.sin(self.theta) * np.sin(self.psi) - np.sin(self.phi) * np.cos(self.psi),
              np.cos(self.phi) * np.cos(self.theta)]])
        R_g_b = R_b_g.T

        u_dot = self.v * self.r - self.w * self.q + g * R_g_b[0, 2] + (X + eta * self.Tmax) / self.m
        v_dot = self.w * self.p - self.u * self.r + g * R_g_b[1, 2] + Y / self.m
        w_dot = self.u * self.q - self.v * self.p + g * R_g_b[2, 2] + Z / self.m

        self.u = self.u + u_dot * dt
        self.v = self.v + v_dot * dt
        self.w = self.w + w_dot * dt

        self.alpha = np.arctan(self.w / self.u)
        self.beta = np.arcsin(self.v / np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2))

        vxyz = R_b_g @ np.array([[self.u], [self.v], [self.w]])
        self.v_x = vxyz[0].item()
        self.v_y = vxyz[1].item()
        self.v_z = vxyz[2].item()

        self.x_g = self.x_g + self.v_x * dt
        self.y_g = self.y_g + self.v_y * dt
        self.z_g = self.z_g + self.v_z * dt

        h_err = -self.z_g - self.h_cmd
        v_x_err = self.v_x - self.v_x_cmd
        v_y_err = self.v_y - self.v_y_cmd
        v_z_err = self.v_z - self.v_z_cmd
        psi_err = self.psi - np.arctan2(self.v_y_cmd, self.v_x_cmd)
        self.h_err_sum = self.h_err_sum + h_err * dt
        self.v_x_err_sum = self.v_x_err_sum + v_x_err * dt
        self.v_y_err_sum = self.v_y_err_sum + v_y_err * dt
        self.psi_err_sum = self.psi_err_sum + psi_err * dt

        state = [h_err, v_x_err, v_y_err, v_z_err, self.phi, self.theta, psi_err, self.p, self.q, self.r,
                 self.h_err_sum, self.v_x_err_sum, self.v_y_err_sum, self.psi_err_sum]

        # state = [self.x_g, self.y_g, self.z_g, self.v_x, self.v_y, self.v_z, self.phi, self.theta, self.psi, self.p,
        #          self.q, self.r, h_err, v_x_err]

        # (0.99**self.t) *
        reward = (0.01 * (abs(h_err) + abs(v_x_err) + abs(v_y_err) + abs(v_z_err)) + 2 * (
                abs(self.beta) + abs(self.phi) + abs(psi_err)) + (
                          abs(self.p) + abs(self.q) + abs(self.r)) + 0.05 * (
                          abs(self.h_err_sum) + abs(self.v_x_err_sum) + abs(self.v_y_err_sum) + abs(
                      self.psi_err_sum)))

        if self.alpha >= 23:
            reward -= 100
        if self.z_g >= -5:
            reward += -100
        if self.z_g > 5:
            done = 1.0

        # reward += self.t

        # for i in range(0, 101, 5):
        #     if self.t == i:
        #         reward += 10
        #
        # for i in range(100, 1001, 100):
        #     if self.t == i:
        #         reward += 100
        #
        # for i in range(-25, -10, 5):
        #     if i+0.5 >= self.z_g >= i-0.5:
        #         reward += 50
        # for i in range(-50, -25, 5):
        #     if i+0.5 >= self.z_g >= i-0.5:
        #         reward += 100
        # for i in range(-100, -50, 5):
        #     if i+0.5 >= self.z_g >= i-0.5:
        #         reward += 200

        # reward = -(0.01 * (abs(h_err) + abs(v_x_err) + abs(v_y_err) + abs(v_z_err)) + 2 * (
        #             abs(self.beta) + abs(self.phi) + abs(psi_err)) + (abs(self.p) + abs(self.q) + abs(self.r)))
        #
        # if abs(self.x_g) > 100 or abs(self.y_g) > 100 or abs(self.theta) >= 89.9 or abs(
        #         self.phi) >= 89.9 or self.alpha < -5 or self.alpha > 25:
        #     # reward += -10
        #     done = 1.0
        # elif self.z_g >= 5 or self.z_g <= -300:
        #     # reward += -1000
        #     if self.t < 50:
        #         reward += -100
        #     done = 1.0
        # elif self.t == 50000:
        #     reward += 1000
        #     done = 1.0
        # reward = -abs(0.01 * h_err) + 0.618

        if self.t == 500000:
            done = 1.0

        return state, reward, done
