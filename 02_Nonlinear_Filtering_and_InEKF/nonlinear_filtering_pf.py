import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag


#############################################################################
#                    TODO: Implement your code here                         #
#############################################################################
def process_model(x, w):
    """
    @param  x:      point with respect to frame 1 (3,)
    @param  w:      noise vector sampled as part of the motion model (3, 1)
    @return x_pred: predicted state vector (3, 1)
    """
    x_pred = np.zeros((3,1)) # placeholder
    x_pred = x.reshape((3,1)) + w
    return x_pred

def measurement_model_1(K_f, p, c):
    """
    @param  K_f: camera intrinisic matrix (2, 2)
    @param  p:   point (3,)
    @param  c:   optical center (2, 1)
    @return z:   measurement (2, 1)
    """
    z = np.zeros((2,1)) # placeholder
    p = p.reshape((3,1))
    pi = 1/p[-1]
    z = (K_f * pi) @ p[:-1] + c
    return z

def measurement_model_2(K_f, p, c, R, t):
    """
    @param  K_f: camera intrinisic matrix (2, 2)
    @param  p:   point (3,)
    @param  c:   optical center (2, 1)
    @param  R:   rotation matrix of camera 2 wrt camera 1 (3, 3)
    @param  t:   translation vector of camera 2 wrt camera 1 (3, 1)
    @return z:   measurement (2, 1)
    """
    z = np.zeros((2,1)) # placeholder
    p = p.reshape((3,1))
    p2 = R.T @ p - R.T @ t
    pi = 1 / p2[-1]
    z = (K_f * pi) @ p2[:-1] + c
    return z

#############################################################################
#                            END OF YOUR CODE                               #
#############################################################################
class particle_filter:
    """
    Particle filter class for state estimation of a nonlinear system
    The implementation follows the Sample Importance Resampling (SIR)
    filter a.k.a bootstrap filter
    """

    def __init__(self, data, N=1000):

        # Constants
        self.R1 = np.cov(data['z_1'], rowvar=False)  # measurement noise covariance
        self.R2 = np.cov(data['z_2'], rowvar=False)  # measurement noise covariance
        self.N  = N # number of particles
        self.Q = np.array([[0.03,0.02,0.01],
                            [0.02,0.04, 0.01],
                            [0.01,0.01, 0.05]]).reshape((3,3)) # input noise covariance (vibrations)
        self.LQ = np.linalg.cholesky(self.Q)
        self.C_1  = data['C_1']
        self.C_2  = data['C_2']
        self.Kf_1 = data['Kf_1']
        self.Kf_2 = data['Kf_2']
        self.Rot  = data['R']
        self.t    = data['t']
        
        # stack the noise covariances for part b
        self.R_stack = block_diag(self.R1, self.R2)

        # Functions
        self.f  = process_model        # process model
        self.h1 = measurement_model_1  # measurement model
        self.h2 = measurement_model_2  # measurement model

        # initialize particles
        self.p = {}
        self.p['x'] = np.zeros((self.N, 3))       # particles
        self.p['w'] = 1/self.N * np.ones(self.N)  # importance weights

        # initial state
        self.x_init  = np.array([0.12, 0.09, 0.5]) # state vector
        self.Sigma_init = np.eye(3)                # state covariance
        self.N_eff = 0

        L_init = np.linalg.cholesky(self.Sigma_init)
        for i in range(self.N):
            self.p['x'][i] = (L_init @ randn(3, 1)).squeeze() + self.x_init

    
    def resampling(self):
        """
        low variance resampling
        """
        W = np.cumsum(self.p['w'])
        r = rand(1) / self.N
        j = 1
        for i in range(self.N):
            u = r + (i - 1) / self.N
            while u > W[j]:
                j = j + 1
            self.p['x'][i, :] = self.p['x'][j, :]
            self.p['w'][i] = 1 / self.N

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    def sample_motion(self):
        """
        random walk motion model
        @note: Use `self.f` instead of `process_model`
        """
        for i in range(self.N):
            # sample noise
            # do not change this line
            w = self.LQ @ randn(3, 1)

            # propagate the particle
            self.p['x'][i, :] = self.f(self.p['x'][i, :], w).squeeze()

    def importance_measurement_1(self, z):
        """
        @param z: stacked measurement vector 2x1
        @note: Use `self.h1` instead of `measurement_model_1`
        """
        # compute importance weights
        w = np.zeros(self.N)
        for i in range(self.N):
            # compute innovation
            z_hat = self.h1(self.Kf_1, self.p['x'][i, :], self.C_1)
            v = z - z_hat
            w[i] = multivariate_normal.pdf(v.flatten(), mean=np.zeros(2), cov=self.R1)

        # update and normalize weights
        self.p['w'] = w / np.sum(w)

        # compute effective number of particles
        self.N_eff = 1 / np.sum(self.p['w']**2)

    def importance_measurement_2(self, z):
        """
        @param z: stacked measurement vector 2x1
        @note: Use `self.h2` instead of `measurement_model_2`
        """
        # compute importance weights
        w = np.zeros(self.N)
        for i in range(self.N):
            # compute innovation
            z_hat = self.h2(self.Kf_2, self.p['x'][i, :], self.C_2, self.Rot, self.t)
            v = z - z_hat
            w[i] = multivariate_normal.pdf(v.flatten(), mean=np.zeros(2), cov=self.R2)

        # update and normalize weights
        self.p['w'] = w / np.sum(w)

        # compute effective number of particles
        self.N_eff = 1 / np.sum(self.p['w']**2)

    def importance_mesurement_batch(self, z):
        """
        @note: should be importance_me*a*surement_batch, for
               compatibility reasons this will not be fixed in
               this semester
        @param z: stacked measurement vector 4x1
        @note: Use `self.h1` instead of `measurement_model_1`
        @note: Use `self.h2` instead of `measurement_model_2`
        """
        # compute importance weights
        w = np.zeros(self.N)
        for i in range(self.N):
            # compute innovation
            z1_hat = self.h1(self.Kf_1, self.p['x'][i, :], self.C_1)
            z2_hat = self.h2(self.Kf_2, self.p['x'][i, :], self.C_2, self.Rot, self.t)
            z_hat_stack = np.vstack((z1_hat, z2_hat)).reshape((4,1))
            v = z - z_hat_stack
            w[i] = multivariate_normal.pdf(v.flatten(), mean=np.zeros(4), cov=self.R_stack)

        # update and normalize weights
        self.p['w'] = w / np.sum(w)

        # compute effective number of particles
        self.N_eff = 1 / np.sum(self.p['w']**2)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


def pf_load_data():
    data = {}
    data['C_1'] = np.loadtxt(open('data/C_1.csv'), delimiter=",").reshape(-1, 1)
    data['C_2'] = np.loadtxt(open('data/C_2.csv'), delimiter=",").reshape(-1, 1)
    data['Kf_1'] = np.loadtxt(open('data/Kf_1.csv'), delimiter=",")
    data['Kf_2'] = np.loadtxt(open('data/Kf_2.csv'), delimiter=",")
    data['R'] = np.loadtxt(open('data/R.csv'), delimiter=",")
    data['t'] = np.loadtxt(open('data/t.csv'), delimiter=",").reshape(-1, 1)
    data['z_1'] = np.loadtxt(open('data/z_1.csv'), delimiter=",")
    data['z_2'] = np.loadtxt(open('data/z_2.csv'), delimiter=",")    
    return data

def pf_sequential(pf, data):
    z_1 = data['z_1']
    z_2 = data['z_2']
    N = np.shape(z_1)[0]
    states = np.zeros((N+1, 3))
    N_t = pf.N/100 # resampling threshold

    states[0] = pf.x_init.squeeze()
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    for i in range(N):
        
        pf.sample_motion()

        pf.importance_measurement_1(z_1[i].reshape((2,1)))
        # use N_t to determine when to resample
        if pf.N_eff < N_t:
            pf.resampling()

        pf.importance_measurement_2(z_2[i].reshape((2,1)))
        # use N_t to determine when to resample
        if pf.N_eff < N_t:
            pf.resampling()

        states[i+1] = np.sum(pf.p['x'] * pf.p['w'].reshape(-1,1), axis=0)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return states

def pf_batch(pf, data):
    z_1 = data['z_1']
    z_2 = data['z_2']
    N = np.shape(z_1)[0]
    states = np.zeros((N+1, 3))
    N_t = pf.N/5 # resampling threshold

    states[0] = pf.x_init.squeeze()
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    for i in range(N):
        
        pf.sample_motion()
        z_stack = np.vstack((z_1[i].reshape((2,1)), z_2[i].reshape((2,1))))
        pf.importance_mesurement_batch(z_stack)
        # use N_t to determine when to resample
        if pf.N_eff < N_t:
            pf.resampling()

        states[i+1] = np.sum(pf.p['x'] * pf.p['w'].reshape(-1,1), axis=0)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
            
    return states
