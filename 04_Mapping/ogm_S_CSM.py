# Occupancy Grid Mapping Counting Sensor Model Class
#
# Author: Chien Erh Lin, Fangtong Liu, Peter Stratton
# Date: 03/09/2024

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

def S_CSM(map, m_i, z_max, w_obstacle, w_beam, num_classes, z, i):
    bearing_diff = []
    # find the nearest beam
    bearing_diff = np.abs(wrapToPI(z[:, 1] - m_i['phi']))
    k = np.nanargmin(bearing_diff)
    bearing_min = bearing_diff[k]
    # -----------------------------------------------
    # To Do: 
    # implement the counting sensor model, update map['alpha']
    # Hint: the way to determine occupied or free is similar to
    # inverse sensor model
    # -----------------------------------------------
    if (m_i['range'] > min(z_max, z[k, 0] + w_obstacle/2)) or (bearing_min > w_beam/2):
        pass
    elif (z[k,0] < z_max) and (np.abs(m_i['range'] - z[k,0]) < w_obstacle/2):
        map['alpha'][i, int(z[k, 2] - 1)] = map['alpha'][i, int(z[k, 2] - 1)] + 1
    elif (m_i['range'] < z[k,0]) and (z[k,0] < z_max):
        map['alpha'][i, num_classes] = map['alpha'][i, num_classes] + 1
    return map


# Occupancy Grid Mapping Class
class ogm_S_CSM:

    def __init__(self, grid_size = 0.135):
        # map dimensions
        self.range_x = [-15, 20]
        self.range_y = [-25, 10]

        # senesor parameters
        self.z_max = 30     # max range in meters
        self.n_beams = 133  # number of beams, we set it to 133 because not all measurements in the dataset contains 180 beams 

        # grid map parameters
        self.grid_size = grid_size
        self.w_obstacle = 2 * self.grid_size    # width of obstacle, 2 * grid_siz
        self.w_beam = 2 * np.pi / self.n_beams  # width of beam, 2 * pi/n_beams
        self.nn = 16                            # number of nearest neighbor search

        # semantic csm function
        self.S_CSM = S_CSM

        # map structure
        self.map = {}  # map
        self.pose = {}  # pose data
        self.scan = []  # laser scan data
        self.m_i = {}  # cell i

        # semantic
        self.num_classes = 6

        # -----------------------------------------------
        # To Do: 
        # prior initialization
        # Initialize prior, prior_alpha
        # -----------------------------------------------
        self.prior = 0.00000001            # prior for setting up mean and variance
        self.prior_alpha = 0.00000001      # a small, uninformative prior for setting up alpha

    def construct_map(self, pose, scan):
        # class constructor
        # construct map points, i.e., grid centroids
        x = np.arange(self.range_x[0], self.range_x[1]+self.grid_size, self.grid_size)
        y = np.arange(self.range_y[0], self.range_y[1]+self.grid_size, self.grid_size)
        X, Y = np.meshgrid(x, y)
        t = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

        # a simple KDTree data structure for map coordinates
        self.map['occMap'] = KDTree(t)
        self.map['size'] = t.shape[0]

        # set robot pose and laser scan data
        self.pose['x'] = pose['x'][0][0]
        self.pose['y'] = pose['y'][0][0]
        self.pose['h'] = pose['h'][0][0]
        self.pose['mdl'] = KDTree(np.hstack((self.pose['x'], self.pose['y'])))
        self.scan = scan

        # -----------------------------------------------
        # To Do: 
        # Initialization map parameters such as map['mean'], map['variance'], map['alpha']
        # -----------------------------------------------
        self.map['mean'] = np.array([self.prior] * self.map['size'] * (self.num_classes + 1)).reshape(self.map['size'], self.num_classes + 1)        # size should be (number of data) x (number of classes + 1)
        self.map['variance'] = np.array([self.prior] * self.map['size']).reshape(-1, 1)    # size should be (number of data) x (1)
        self.map['alpha'] = np.array([self.prior_alpha] * self.map['size'] * (self.num_classes + 1)).reshape(self.map['size'], self.num_classes + 1)       # size should be (number of data) x (number of classes + 1)

    def is_in_perceptual_field(self, m, p):
        # check if the map cell m is within the perception field of the
        # robot located at pose p
        inside = False
        d = m - p[0:2].reshape(-1)
        self.m_i['range'] = np.sqrt(np.sum(np.power(d, 2)))
        self.m_i['phi'] = wrapToPI(np.arctan2(d[1], d[0]) - p[2])
        # check if the range is within the feasible interval
        if (0 < self.m_i['range']) and (self.m_i['range'] < self.z_max):
            # here sensor covers -pi to pi
            if (-np.pi < self.m_i['phi']) and (self.m_i['phi'] < np.pi):
                inside = True
        return inside

    def build_ogm(self, disable_print=False):
        # build occupancy grid map using the binary Bayes filter.
        # We first loop over all map cells, then for each cell, we find
        # N nearest neighbor poses to build the map. Note that this is
        # more efficient than looping over all poses and all map cells
        # for each pose which should be the case in online (incremental)
        # data processing.
        for i in tqdm(range(self.map['size']), disable=disable_print):
            m = self.map['occMap'].data[i, :]
            _, idxs = self.pose['mdl'].query(m, self.nn)
            if len(idxs):
                for k in idxs:
                    # pose k
                    pose_k = np.array([self.pose['x'][k], self.pose['y'][k], self.pose['h'][k]])
                    if self.is_in_perceptual_field(m, pose_k):
                        # laser scan at kth state; convert from cartesian to
                        # polar coordinates
                        z = cart2pol(self.scan[k][0][0, :], self.scan[k][0][1, :])
                        z = np.hstack((z[:,0].reshape(-1, 1), z[:,1].reshape(-1, 1), self.scan[k][0][2, :].reshape(-1, 1).astype(int)))
                        # -----------------------------------------------
                        # To Do: 
                        # update the sensor model in cell i
                        # -----------------------------------------------
                        self.S_CSM(self.map, self.m_i, self.z_max, self.w_obstacle, self.w_beam, self.num_classes, z, i)

            # -----------------------------------------------
            # To Do: 
            # update mean and variance for each cell i
            # -----------------------------------------------
            alpha_ik_sum = np.sum(self.map['alpha'][i, :])
            self.map['mean'][i] = self.map['alpha'][i] / alpha_ik_sum
            self.map['variance'][i] = (max(self.map['alpha'][i]) / alpha_ik_sum) * (1 - (max(self.map['alpha'][i]) / alpha_ik_sum)) / (alpha_ik_sum + 1)
    

# This function is used to convert Cartesian to Polar
def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    z = np.hstack((r.reshape(-1, 1), theta.reshape(-1, 1)))
    return z


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def wrapToPI(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    idx = np.argwhere(np.abs(x_wrap) > np.pi)
    while len(idx):
        x_wrap[idx] -= 2 * np.pi * np.sign(x_wrap[idx])
        idx = np.argwhere(np.abs(x_wrap) > np.pi)
    return x_wrap