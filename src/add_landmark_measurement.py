import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    pose_4 = result.atPose2(X(4))          # get the optimized pose of X(4)
    landmark_2 = result.atPoint2(L(2))     # get the optimized position of L(2)

    # Vector from robot to landmark in global frame
    dx = landmark_2[0] - pose_4.x()
    dy = landmark_2[1] - pose_4.y()

    # Global angle to landmark
    global_angle_deg = np.degrees(np.arctan2(dy, dx))

    # Bearing = angle relative to robot heading
    rotation = global_angle_deg - np.degrees(pose_4.theta())
    distance = 8 - (4*np.sqrt(2))
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph