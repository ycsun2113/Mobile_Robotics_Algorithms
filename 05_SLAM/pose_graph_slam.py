import gtsam.noiseModel
import numpy as np
import gtsam
from typing import NamedTuple

def pretty_print(arr):
    return '\n'.join([' '.join(['%.2f' % x for x in c]) for c in arr])

class Pose2(NamedTuple):
    '''
    Pose2 class for 2D pose
    @usage: pose = Pose2(id, x, y, z)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    theta: float

class Edge2(NamedTuple):
    '''
    Edge2 class for 2D edge
    @usage: edge = Edge2(id1, id2, x, y, z, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    theta: float
    info: np.ndarray # 3x3 matrix

    def __str__(self):
        return f"Edge2(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, theta={self.theta},\ninfo=\n{pretty_print(self.info)})\n"

class Pose3(NamedTuple):
    '''
    Pose3 class for 3D pose
    @usage: pose = Pose3(id, x, y, z, qx, qy, qz, qw)
            print(pose.x)
    '''
    id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

class Edge3(NamedTuple):
    '''
    Edge3 class for 3D edge
    @usage: edge = Edge3(id1, id2, x, y, z, qx, qy, qz, qw, info)
            print(edge.x)
    '''
    id1: int
    id2: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    info: np.ndarray # 6x6 matrix

    def __str__(self):
        return f"Edge3(id1={self.id1}, id2={self.id2}, x={self.x}, y={self.y}, z={self.z}, qx={self.qx}, qy={self.qy}, qz={self.qz}, qw={self.qw},\ninfo=\n{pretty_print(self.info)})\n"


def read_g2o_2d(file_name):
    data = {
        'poses': [],
        'edges': []
    }

    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose2 or Edge2 objects
        
        for line in lines:
            elements = line.strip().split()

            # pose
            if elements[0] == "VERTEX_SE2":
                id = int(elements[1])
                x, y, theta = map(float, elements[2:5])
                data['poses'].append(Pose2(id, x, y, theta))

            # edge
            elif elements[0] == "EDGE_SE2":
                id1 = int(elements[1])
                id2 = int(elements[2])
                x, y, theta = map(float, elements[3:6])
                info = list(map(float, elements[6:]))
                info_mat = np.array([[info[0], info[1], info[2]], 
                                     [info[1], info[3], info[4]],
                                     [info[2], info[4], info[5]]])
                cov = np.linalg.inv(info_mat)
                data['edges'].append(Edge2(id1, id2, x, y, theta, info_mat))

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    return data

def gn_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # set initial_values according to poses
    for pose in poses:
        id, x, y, theta = pose
        initial_values.insert(id, gtsam.Pose2(x, y, theta))

    # add prior factor for the first pose
    prior_id = poses[0].id
    prior_mean = gtsam.Pose2(poses[0].x, poses[0].y, poses[0].theta)
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([first_pose_prior_cov[0], first_pose_prior_cov[1], first_pose_prior_cov[2]]))
    graph.add(gtsam.PriorFactorPose2(prior_id, prior_mean, prior_noise))

    # add between factors according to edges
    for edge in edges:
        id1, id2, dx, dy, dtheta, info_mat = edge
        cov = np.linalg.inv(info_mat)
        edge_noise = gtsam.noiseModel.Gaussian.Covariance(cov)
        graph.add(gtsam.BetweenFactorPose2(id1, id2, gtsam.Pose2(dx, dy, dtheta), edge_noise))

    # optimize the graph
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def isam_2d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()
    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    isam = gtsam.ISAM2()

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if frame_id==0:
            # initialization
            prior_pose = gtsam.Pose2(pose.x, pose.y, pose.theta)
            prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([first_pose_prior_cov[0], first_pose_prior_cov[1], first_pose_prior_cov[2]]))
            graph.add(gtsam.PriorFactorPose2(frame_id, prior_pose, prior_noise))
            initial_values.insert(frame_id, prior_pose)
            # pass
        else:
            # optimize new frame
            prevPose = result.atPose2(frame_id - 1)
            initial_values.insert(frame_id, prevPose)

            for edge in edges:
                id1, id2, dx, dy, dtheta, info_mat = edge
                if id2 == frame_id:
                    cov = np.linalg.inv(info_mat)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.BetweenFactorPose2(id1, id2, gtsam.Pose2(dx, dy, dtheta), model))

            # pass

        # update isam
        isam.update(graph, initial_values)
        result = isam.calculateEstimate()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose2(result)

def read_g2o_3d(file_name):
    data = {
        'poses': [],
        'edges': []
    }


    # read the file
    with open(file_name, "r") as f:
        lines = f.readlines()

        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################

        # fill in the `data` dict with Pose3 or Edge3 objects

        for line in lines:
            elements = line.strip().split()

            # pose
            if elements[0] == "VERTEX_SE3:QUAT":
                id = int(elements[1])
                x, y, z, qx, qy, qz, qw = map(float, elements[2:])
                data['poses'].append(Pose3(id, x, y, z, qx, qy, qz, qw))

            # edge
            elif elements[0] == "EDGE_SE3:QUAT":
                id1 = int(elements[1])
                id2 = int(elements[2])
                x, y, z, qx, qy, qz, qw = map(float, elements[3:10])
                info = list(map(float, elements[10:]))
                info_mat = np.array([[info[0], info[1],  info[2],  info[3],  info[4],  info[5]], 
                                     [info[1], info[6],  info[7],  info[8],  info[9],  info[10]],
                                     [info[2], info[7],  info[11], info[12], info[13], info[14]],
                                     [info[3], info[8],  info[12], info[15], info[16], info[17]],
                                     [info[4], info[9],  info[13], info[16], info[18], info[19]],
                                     [info[5], info[10], info[14], info[17], info[19], info[20]]])
                data['edges'].append(Edge3(id1, id2, x, y, z, qx, qy, qz, qw, info_mat))

        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
    
    return data  

def gn_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # set initial_values according to poses
    for pose in poses:
        id, x, y, z, qx, qy, qz, qw = pose
        initial_values.insert(id, gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(x, y, z)))

    # add prior factor for the first pose
    prior_id = poses[0].id
    pose3_rot = gtsam.Rot3.Quaternion(poses[0].qw, poses[0].qx, poses[0].qy, poses[0].qz)
    pose3_point = gtsam.Point3(poses[0].x, poses[0].y, poses[0].z)
    prior_mean = gtsam.Pose3(pose3_rot, pose3_point)
    prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([first_pose_prior_cov[0], first_pose_prior_cov[1], first_pose_prior_cov[2], first_pose_prior_cov[3], first_pose_prior_cov[4], first_pose_prior_cov[5]]))
    graph.add(gtsam.PriorFactorPose3(prior_id, prior_mean, prior_noise))

    # add between factors according to edges
    for edge in edges:
        id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw, info_mat = edge
        edge_rot = gtsam.Rot3.Quaternion(dqw, dqx, dqy, dqz)
        edge_trans = gtsam.Point3(dx, dy, dz)
        cov = np.linalg.inv(info_mat)
        edge_noise = gtsam.noiseModel.Gaussian.Covariance(cov)
        graph.add(gtsam.BetweenFactorPose3(id1, id2, gtsam.Pose3(edge_rot, edge_trans), edge_noise))

    # optimize the graph
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_values)
    result = optimizer.optimize()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)

def isam_3d(data):
    poses = data['poses']
    edges = data['edges']
    result = gtsam.Values()

    # use this covariance for the prior factor of the first pose
    first_pose_prior_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # create optimizer
    isam = gtsam.ISAM2()

    for pose in poses:

        frame_id = pose.id

        # create an empty factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        if frame_id==0:
            # initialization

            # Notice that the order of quaternion is different from
            # the one in the g2o file. GTSAM uses (qw, qx, qy, qz).

            pose3_rot = gtsam.Rot3.Quaternion(pose.qw, pose.qx, pose.qy, pose.qz)
            pose3_point = gtsam.Point3(pose.x, pose.y, pose.z)
            prior_pose = gtsam.Pose3(pose3_rot, pose3_point)
            prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([first_pose_prior_cov[0], first_pose_prior_cov[1], first_pose_prior_cov[2], first_pose_prior_cov[3], first_pose_prior_cov[4], first_pose_prior_cov[5]]))
            graph.add(gtsam.PriorFactorPose3(frame_id, prior_pose, prior_noise))
            initial_values.insert(frame_id, prior_pose)
            # pass
        else:
            # optimize new frame
            prevPose = result.atPose3(frame_id - 1)
            initial_values.insert(frame_id, prevPose)

            for edge in edges:
                id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw, info_mat = edge
                if id2 == frame_id:
                    edge_rot = gtsam.Rot3.Quaternion(dqw, dqx, dqy, dqz)
                    edge_trans = gtsam.Point3(dx, dy, dz)
                    cov = np.linalg.inv(info_mat)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.BetweenFactorPose3(id1, id2, gtsam.Pose3(edge_rot, edge_trans), model))
            # pass

        # update isam
        isam.update(graph, initial_values)
        result = isam.calculateEstimate()

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    # return the poses
    return gtsam.utilities.extractPose3(result)
