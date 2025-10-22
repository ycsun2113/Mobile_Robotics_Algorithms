import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u, X, P, step):
        if step == 0:
            X = self.state_.getState()
            P = self.state_.getCovariance()
        else:
            X = X
            P = P
        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################

        # Jacobian of motion model w.r.t location
        G = self.Gfun(X, u)

        # Jacobian of the motion model w.r.t control
        V = self.Vfun(X, u)

        # Motion noise
        M = self.M(u)

        # Predicted mean
        X_pred = self.gfun(X, u)

        # Predicted covariance
        P_pred = G @ P @ G.T + V @ M @ V.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(X_pred), np.copy(P_pred)


    def correction(self, z, landmarks, X, P):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = X
        P_predict = P
        
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

        # Predicted measurement mean
        landmark1_x = landmark1.getPosition()[0]
        landmark1_y = landmark1.getPosition()[1]
        landmark2_x = landmark2.getPosition()[0]
        landmark2_y = landmark2.getPosition()[1]

        z_hat1 = self.hfun(landmark1_x, landmark1_y, X_predict)
        z_hat2 = self.hfun(landmark2_x, landmark2_y, X_predict)
        z_hat = np.hstack((z_hat1, z_hat2))
        # print("z_meas size", z_meas.shape)
        # print("z_hat size", z_hat.shape)

        # Jacobian of h w.r.t location
        H1 = self.Hfun(landmark1_x, landmark1_y, X_predict, z_hat1)
        H2 = self.Hfun(landmark2_x, landmark2_y, X_predict, z_hat2)
        H = np.vstack((H1, H2))
        # print("H_1 size", H1.shape)
        # print("H size", H.shape)
        # print("P_predict size", P_predict.shape)

        # Measurement covariance
        Q = block_diag(self.Q, self.Q)
        # print("Q size", Q.shape)

        # Innovation covariance
        S = H @ P_predict @ H.T + Q

        # Kalman gain
        K = P_predict @ H.T @ np.linalg.inv(S)

        # Updated mean
        z_delta = z_meas - z_hat
        z_delta[0] = wrap2Pi(z_delta[0])
        z_delta[2] = wrap2Pi(z_delta[2])
        X = X_predict + K @ z_delta

        # Updated covariance
        I = np.eye(P_predict.shape[0])
        KH = K @ H
        P = (I - KH) @ P_predict

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state