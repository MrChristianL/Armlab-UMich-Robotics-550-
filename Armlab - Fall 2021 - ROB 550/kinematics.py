"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from copy import deepcopy


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    T = np.eye(4)
    for i in range(0, link):
        dh_param = deepcopy(dh_params[i,:])
        
        if i == 0 or i==2 or i==3 or i==4:
            dh_param[3] += joint_angles[i]
        elif i==1:
            dh_param[3] -= joint_angles[i]

        T = np.matmul(
                    T,
                    get_transform_from_dh(dh_param[0], dh_param[1], dh_param[2], dh_param[3])
                )
    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    
    T = [[np.cos(theta), -(np.sin(theta)*np.cos(alpha)), (np.sin(theta)*np.sin(alpha)), a*np.cos(theta)], 
        [np.sin(theta), (np.cos(theta)*np.cos(alpha)), -(np.cos(theta)*np.sin(alpha)), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
        ]
    return np.array(T)


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the ZYZ euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T [phi, theta, psi].
    
    """
    R = T[0:3, 0:3]  # 3D Rotation 
    
    # print("R", R)
    
    # 3 cases for edge cases
    if R[2,2] > .99999:
        theta = 0
        phi = 0
        psi = np.arctan2(-R[0,1], R[0,0])
    elif R[2,2] < -.99999:
        theta = np.pi
        phi = 0
        psi = -np.arctan2(-R[0,1], -R[0,0])
    else:
        theta = np.arctan2(np.sqrt(1-R[2,2]*R[2,2]), R[2,2])
        phi = np.arctan2(R[1,2], R[0,2])
        psi = np.arctan2(R[2,1],-R[2,0])

    return np.array([phi, theta, psi]) 


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO [N]: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pose = np.zeros(4)
    
    pose[0:3] = T[0:3,3]  # translation

    pose[3] = get_euler_angles_from_T(T)[1] - np.pi/2  # angle is second element of ZYZ euler angle
    
    return pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    dh_params = deepcopy(dh_params)
    p_0 = np.array(pose[:3]).reshape(3,1)
    p_0 = np.vstack((p_0,[1])) # desired point in coordinate frame 0 (world frame)
    phi = pose[3]  # desired angle of attack to pick up block

    # theta1 turn towards point
    if abs(p_0[0,0]) < 1E-4 and abs(p_0[1,0]) < 1E-4:
        theta1 = 0.0
    else:
        theta1 = np.arctan2(p_0[1,0],p_0[0,0]) - np.pi/2


    # theta2 and theta3 is planar 2D arm
    T1_0 = np.linalg.inv(FK_dh(dh_params, [theta1], 1))
    p1 = np.matmul(T1_0, p_0) 
    
    l1 = dh_params[1,0]  # DH parameter "a" for link 2 (205.73)
    l2 = dh_params[2,0]  # DH parameter "a" for link 3 (200)24tr23r
    d5 = dh_params[4,2]  # DH parameter "d" for link 5 (160)

    x_3 = -p1[0,0] - d5*np.cos(phi)  # The origin of frame 3 with respect to negative of x1,y1
    y_3 = -p1[1,0] + d5*np.sin(phi)  

    theta3_before = -np.arccos(((x_3**2 + y_3**2) - l1**2 -l2**2)/(2*l1*l2))  # negative because that's the elbow up position
    theta2_before = np.arctan2(y_3,x_3) - np.arctan2(l2*np.sin(theta3_before), l1+l2*np.cos(theta3_before))
    
    theta3 = theta3_before - dh_params[2,3]  #+ 1.326   # correcting for rotation of the axes
    theta2 = np.pi - theta2_before + dh_params[1,3] #-1.816 # correcting for rotation of the axes

    # theta4 is the negative of angle of attack
    theta4 = -(theta2_before + theta3_before) - phi

    # theta5 will be set later based on block orientation, but the default of 0 should work in most cases
    theta5 = 0

    theta1 = clamp(float(theta1))
    theta2 = clamp(float(theta2))
    theta3 = clamp(float(theta3))
    theta4 = clamp(float(theta4))

    return [np.array([theta1, theta2, theta3, theta4, theta5])]