#!/usr/bin/python
"""!
Test kinematics

TODO: Use this file and modify as you see fit to test kinematics.py
"""
import argparse
import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
from kinematics import *
from config_parse import *
from copy import deepcopy

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--dhconfig", required=True, help="path to DH parameters csv file")

    args=vars(ap.parse_args())
    #args = {'dhconfig':'/home/student/Desktop/Team 107/armlab/config/rx200_dh.csv '}  # bypass arg parser

    passed = True
    vclamp = np.vectorize(clamp)

    dh_params = parse_dh_param_file(args['dhconfig'])

    print('Test get_euler_angles_from_T')
    T0 = np.eye(4)
    T1 = FK_dh(deepcopy(dh_params), [0,0,0,0], 1)
    T2 = FK_dh(deepcopy(dh_params), [0,0,0,0], 3)
    out0 = get_euler_angles_from_T(T0)
    out1 = get_euler_angles_from_T(T1)
    out2 = get_euler_angles_from_T(T2)
    # print("out0 = ")
    # print(out0)
    # print("out1 = ")
    # print(out1)
    # print("out2 = ")
    # print(out2)
    assert(all(abs(angle)<1E-4 for angle in out0))  # all angles should be 0
    assert(abs(out1[0])<1E-4)
    assert(abs(out1[1]-np.pi/2)<1E-4)
    assert(abs(out1[2] - (-np.pi/2))<1E-4)

    print('Test get_pose_from_T')
    T0 = np.eye(4)
    T1 = FK_dh(deepcopy(dh_params), [0,0,0,0], 1)
    T2 = FK_dh(deepcopy(dh_params), [0,0,0,0], 3)
    out0 = get_pose_from_T(T0)
    out1 = get_pose_from_T(T1)
    out2 = get_pose_from_T(T2)
    assert(abs(out0[0])<1E-4)
    assert(abs(out0[1])<1E-4)
    assert(abs(out0[2])<1E-4)
    assert(abs(out0[3] - (-np.pi/2))<1E-4)
    assert(abs(out1[0])<1E-4)
    assert(abs(out1[1])<1E-4)
    assert(abs(out1[2] - (103.91))<1E-4)
    assert(abs(out1[3])<1E-4)
    print("\n\n")


    ### Add arm configurations to test here
    fk_angles = [
                #  [0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.1, 0.0, 0.0, 0.0, 0.0],
                 [-0.1, 0.0, 0.0, 0.0, 0.0],
                 [1, 0.0, 0.0, 0.0, 0.0],
                 [-1, 0.0, 0.0, 0.0, 0.0],
                #  [0.0, 39.13*(2*np.pi/360), 0.0, 0.0, 0.0],   # this is where theta1 is changed to make gripper touch flat plane
                #  [0.0, 0.0, 0.1, 0.0, 0.0],
                #  [0.0, 0.0, 0.0, 0.1, 0.0],
                #   np.pi/180*np.array([30.41,12.57,-14.41,-48.08,0]),
                #   np.pi/180*np.array([43.86,52.47,51.15,-90.62,0]),
                #   np.pi/180*np.array([-49.57,25.93,-21.09,12.30,0]),
                #   np.pi/180*np.array([.79,-12.22,-52.12,-55.99,0]),
                 ]
    
    print('Test FK')
    fk_poses = []
    for joint_angles in fk_angles:
        print('Joint angles:', joint_angles)
        for i, _ in enumerate(joint_angles):
            pose = get_pose_from_T(FK_dh(deepcopy(dh_params), joint_angles, i+1))
            print('Link {} pose: {}'.format(i+1, pose))
            if i == len(joint_angles) - 1:
                fk_poses.append(pose)
        print()

    print('Test IK')
    for pose, angles in zip(fk_poses, fk_angles):
        matching_angles = False
        print('Pose: {}'.format(pose))
        options = IK_geometric(deepcopy(dh_params), pose)
        for i, joint_angles in enumerate(options):
            print('Option {}: {}'.format(i, joint_angles))
            compare = vclamp(joint_angles - angles)
            if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-3):
                print('Option {} matches angles used in FK'.format(i))
                matching_angles = True
        if not matching_angles:
            print('No match to the FK angles found!')
            passed = False
        print()

    # print('Test IK ONLY')
    # for pose, angles in zip(fk_poses, fk_angles):
    #     matching_angles = False
    #     print('Pose: {}'.format(pose))
    #     options = IK_geometric(deepcopy(dh_params), pose)
    #     for i, joint_angles in enumerate(options):
    #         print('Option {}: {}'.format(i, joint_angles))
    #         compare = vclamp(joint_angles - angles)
    #         if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-3):
    #             print('Option {} matches angles used in FK'.format(i))
    #             matching_angles = True
    #     if not matching_angles:
    #         print('No match to the FK angles found!')
    #         passed = False
    #     print()