import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

def wedge(phi):
    """
    R^3 vector to so(3) matrix
    @param  phi: R^3
    @return Phi: so(3) matrix
    """
    phi = phi.squeeze()
    Phi = np.array([[0, -phi[2], phi[1]],
                    [phi[2], 0, -phi[0]],
                    [-phi[1], phi[0], 0]])
    return Phi

def adjoint(R):
    """
    Adjoint of SO3 Adjoint (R) = R
    """
    return R

#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
def motion_model(R, omega, dt):
    """
    @param  R:      State variable (3, 3)
    @param  omega:  gyroscope reading (3,)
    @param  dt:     time step
    @return R_pred: predicted state variable (3, 3)
    """
    R_pred = np.zeros((3,3)) # placeholder
    R_pred = R @ expm(wedge(omega) * dt)
    return R_pred


def measurement_Jacobain(g):
    """
    @param  g: gravity (3,)
    @return H: measurement Jacobain (3, 3)
    """
    H = np.zeros((3,3)) # placeholder
    # print("g: ", g)
    # H = np.array([[0, -g[2], g[1]], 
    #               [g[2], 0, -g[0]], 
    #               [-g[1], -g[0], 0]])
    H = wedge(g)
    return H

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################

class right_iekf:

    def __init__(self):
        self.Phi = np.eye(3)               # state transtion matrix
        self.Q = 1e-4*np.eye(3)            # gyroscope noise covariance
        self.N = 1e-4*np.eye(3)            # accelerometer noise covariance
        self.f = motion_model              # process model
        self.H = measurement_Jacobain      # measurement Jacobain
        self.X = np.eye(3)                 # state vector
        self.P = 0.1 * np.eye(3)           # state covariance

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    def prediction(self, omega, dt):
        """
        @param omega: gyroscope reading
        @param dt:    time step
        """
        adjoint_X = self.X

        self.X = self.f(self.X, omega, dt)

        # self.P = self.Phi @ self.P + self.P @ self.Phi.T + adjoint_X @ self.Q @ adjoint_X.T
        # self.P = self.Phi @ self.P @ self.Phi + self.Q
        self.P = self.Phi @ self.P @ self.Phi + adjoint_X @ self.Q @ adjoint_X.T

        return

    def correction(self, Y, g):
        """
        @param Y: linear acceleration measurement
        @param g: gravity
        """

        H = self.H(g)
        N = self.X @ self.N @ self.X.T
        S = H @ self.P @ H.T + N
        L = self.P @ H.T @ np.linalg.inv(S)
        I = np.eye(3)

        exp_value = expm(wedge(L @ (self.X @Y -  g)))
        self.X = np.dot(exp_value, self.X)

        # Update Covariance
        temp = I - L @ H
        self.P = temp @ self.P @ temp.T + L @ N @ L.T

        return

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


def riekf_load_data():
    data = {}
    data['accel'] = np.loadtxt(open('data/a.csv'), delimiter=",")
    data['omega'] = np.loadtxt(open('data/omega.csv'), delimiter=",")
    data['dt'] = np.loadtxt(open('data/dt.csv'), delimiter=",")
    data['gravity'] = np.loadtxt(open('data/gravity.csv'), delimiter=",")
    data['euler_gt'] = np.loadtxt(open('data/euler_gt.csv'), delimiter=",")
    return data


def ahrs_riekf(iekf_filter, data):
    # useful variables
    accel = data['accel']
    omega = data['omega']
    dt = data['dt']
    gravity = data['gravity']
    N = data['accel'].shape[0]

    states_rot = np.zeros((N+1, 3, 3))
    states_rot[0] = iekf_filter.X
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    for i in range(N):

        iekf_filter.prediction(omega[i], dt[i])

        Y = accel[i]
        iekf_filter.correction(Y, gravity)

        states_rot[i+1] = iekf_filter.X
        # print(states_rot[i])
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    # convert rotation matrices to euler angles
    states_euler = np.zeros((N+1, 3))
    for i, rot in enumerate(states_rot):
        r = R.from_matrix(rot)
        states_euler[i] = r.as_euler('zyx')
    return states_euler
