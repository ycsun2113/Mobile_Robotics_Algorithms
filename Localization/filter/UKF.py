

from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)
        self.n = 6


    def prediction(self, u, X, P , step):
        # prior belief
        if step == 0:
            mean = self.state_.getState()
            sigma = self.state_.getCovariance()
        else:
            mean = X
            sigma = P
        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################

        # Motion noise
        M = self.M(u)
        # print("X shape: ", mean.shape)
        # print("P shape: ", sigma.shape)
        # print("M shape: ", M.shape)

        # Measurement noise
        Q = self.Q
        # print("Q shape: ", Q.shape)

        # Augmented state mean
        X_aug = np.hstack((mean.reshape((3,1)).T, np.zeros((M.shape[0] + Q.shape[0],1)).T)).T
        # print("X_aug shape: ", X_aug.shape)

        # Augmented covariance
        P_aug = block_diag(sigma, M, Q)
        # print("P_aug shape: ", P_aug.shape)

        # Sigma points
        self.sigma_point(X_aug, P_aug, self.kappa_g)
        X_sigma = self.X[:3, :]
        u_sigma = self.X[3:6, :]
        # print("sigma points (all): ", self.X.shape)
        # print("sigma x: ", X_sigma.shape)
        # print("sigma u: ", u_sigma.shape)
        # print("w: ", self.w.shape)

        # Prediction of sigma points
        X_sigma_pred = np.zeros((3, 2*self.n + 1))
        for i in range(2 * self.n + 1):
            X_sigma_pred[:, i] = self.gfun(X_sigma[:, i], u)
        # print("X_sigma_pred shape: ", X_sigma_pred.shape)

        # Predicted mean
        X_pred = np.zeros((3,1))
        for i in range(2*self.n + 1):
            X_pred += self.w[i] * X_sigma_pred[:, i].reshape(-1,1)
        # print("X_pred shape: ", X_pred.shape)

        # Predicted covariance
        P_pred = np.zeros((3,3))
        for i in range(2*self.n + 1):
            x_delta = X_sigma_pred[:, i].reshape(-1,1) - X_pred[:]
            x_delta[2] = wrap2Pi(x_delta[2])
            P_pred += self.w[i] * (x_delta @ x_delta.T)
        # print("P_pred shape: ", P_pred.shape)

        self.Y = np.copy(X_sigma_pred)
        # print("X_sigma_pred shape: ", X_sigma_pred.shape)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred.reshape(3,1))
        self.state_.setCovariance(P_pred)
        return np.copy(self.Y), np.copy(self.w), np.copy(X_pred), np.copy(P_pred)

    def correction(self, z, landmarks, Y, w, X, P):

        X_predict = X
        P_predict = P
        self.Y = Y
        self.w = w        
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        # extract measurements from z
        z_meas = np.hstack((z[:2], z[3:5]))
        X_sigma_pred = self.Y

        # Measurement sigma points
        landmark1_x = landmark1.getPosition()[0]
        landmark1_y = landmark1.getPosition()[1]
        landmark2_x = landmark2.getPosition()[0]
        landmark2_y = landmark2.getPosition()[1]

        Z_sigma_points_1 = np.zeros((2, 2*self.n + 1))
        Z_sigma_points_2 = np.zeros((2, 2*self.n + 1))
        for i in range(2 * self.n + 1):
            Z_sigma_points_1[:, i] = self.hfun(landmark1_x, landmark1_y, X_sigma_pred[:, i])
            Z_sigma_points_2[:, i] = self.hfun(landmark2_x, landmark2_y, X_sigma_pred[:, i])


        Z_sigma_points = np.vstack((Z_sigma_points_1, Z_sigma_points_2))
        Z_sigma_points[0, :] = wrap2Pi(Z_sigma_points[0, :])
        Z_sigma_points[2, :] = wrap2Pi(Z_sigma_points[2, :])
        
        # Predicted measurement mean
        z_hat = np.zeros((4,1))
        for i in range(2*self.n + 1):
            z_hat[:] += self.w[i] * Z_sigma_points[:, i].reshape(-1,1)
        # print("z_hat shape: ", z_hat.shape)

        # Predicted measurement covariance 
        S = np.zeros((4, 4))

        # Cross-covariance 
        P_xy = np.zeros((3, 4))

        z_delta = np.zeros((4,1))
        X_delta = np.zeros((3,1))
        for i in range(2*self.n + 1):
            z_delta = Z_sigma_points[:, i].reshape(-1,1) - z_hat
            z_delta[0] = wrap2Pi(z_delta[0])
            z_delta[2] = wrap2Pi(z_delta[2])

            X_delta = X_sigma_pred[:, i].reshape(-1,1) - X_predict
            X_delta[2] = wrap2Pi(X_delta[2])

            # Predicted measurement covariance 
            S += self.w[i] * np.outer(z_delta, z_delta)
            # Cross-covariance 
            P_xy += self.w[i] * np.outer(X_delta, z_delta)

        S += block_diag(self.Q, self.Q)

        # Kalman gain
        K = P_xy @ np.linalg.inv(S)
        # print("S shape: ", S.shape)
        # print("P_xy shape: ", P_xy.shape)
        # print("K shape: ", K.shape)

        # Updated mean
        inovation = z_meas.reshape(-1,1) - z_hat
        inovation[0] = wrap2Pi(inovation[0])
        inovation[2] = wrap2Pi(inovation[2])
        X = X_predict + K @ inovation
        X = X.reshape(3,)
        # print("new X: ", X.shape)

        # Updated covariance
        P = P_predict - K @ S @ K.T
        # print("new P: ", P.shape)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        # print("self.n: ", self.n)
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state