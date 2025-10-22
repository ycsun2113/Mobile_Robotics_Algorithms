
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u, Sigma, mu, step):
        if step != 0 :
            self.Sigma = Sigma
            self.mu = mu
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        
        # compute AdjX
        AdjX = np.hstack((self.mu[0:2, 0:2], np.array([[self.mu[1, 2]], [-self.mu[0, 2]]])))
        AdjX = np.vstack((AdjX, np.array([0, 0, 1])))

        # propagation
        self.mu_pred, self.sigma_pred = self.propagation(u_se2, AdjX, self.mu, self.Sigma, self.W)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        return np.copy(self.mu_pred), np.copy(self.sigma_pred)

    def propagation(self, u, adjX, mu, Sigma , W):
        self.mu = mu
        self.Sigma = Sigma
        self.W = W
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################

        # predict mu
        self.mu_pred = self.mu @ expm(u)

        # predict sigma
        self.sigma_pred = self.Sigma + adjX @ self.W @ adjX.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        return np.copy(self.mu_pred), np.copy(self.sigma_pred)
        
    def correction(self, Y1, Y2, z, landmarks, mu_pred, sigma_pred):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        landmark1_x = landmark1.getPosition()[0]
        landmark1_y = landmark1.getPosition()[1]
        landmark2_x = landmark2.getPosition()[0]
        landmark2_y = landmark2.getPosition()[1]

        # find H
        H1 = np.array([[landmark1_x, -1, 0],
                       [-landmark1_y, 0, -1]])
        H2 = np.array([[landmark2_x, -1, 0],
                       [-landmark2_y, 0, -1]])
        H = np.vstack((H1, H2))
        # print("H shape: ", H.shape)

        # find N
        N = np.dot(np.dot(self.mu_pred, block_diag(self.V, 0)), self.mu_pred.T)
        N = block_diag(N[0:2, 0:2], N[0:2, 0:2]) # 4 x 4 block-diagonal matrix
        # print("N shape: ", N.shape)

        # filter gain
        S = np.dot(np.dot(H, self.sigma_pred), H.T) + N
        L = np.dot(np.dot(self.sigma_pred, H.T), np.linalg.inv(S))
        # print("S shape: ", S.shape)
        # print("L shape: ", L.shape)

        # update state
        b1 = np.hstack((landmark1_x, landmark1_y, 1))
        b2 = np.hstack((landmark2_x, landmark2_y, 1))
        # print("b1 shape: ", b1.shape)
        # print("b2 shape: ", b2.shape)
        # print("Y1 shape: ", Y1.shape)
        # print("Y2 shape: ", Y2.shape)
        nu = np.dot(block_diag(self.mu_pred, self.mu_pred), np.vstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))) - np.vstack((b1.reshape(-1, 1), b2.reshape(-1, 1)))
        nu = np.hstack((nu[0:2, 0], nu[3:5, 0]))
        # print("nu shape: ", nu.shape)
        # print("nu shape: ", nu.shape)
        innovation = np.dot(L, nu)
        innovation_wedge = np.array([[0, -innovation[2], innovation[0]],
                                     [innovation[2], 0, innovation[1]],
                                     [0, 0, 0]])
        self.mu = np.dot(expm(innovation_wedge), self.mu_pred)
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        # print("innovation shape", innovation.shape)
        # print("innovation_wedge shape", innovation_wedge.shape)
        # print("mu shape", self.mu.shape)
        # print("X shape: ", X.shape)

        # update covariance
        I = np.eye(np.shape(self.sigma_pred)[0])
        temp = I - np.dot(L, H)
        self.Sigma = np.dot(np.dot(temp, self.sigma_pred), temp.T) + np.dot(np.dot(L, N), L.T)


        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)
        return np.copy(X), np.copy(self.Sigma), np.copy(self.mu)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
