"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import rank, sort
import rospy
import cv2
#import pandas as pd
import kinematics
from rospy.timer import sleep
#import control_station

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.mousexyz_world = np.array([[0,0,0]])  # world coordinates of mouse location
        self.block_pickup_xyz = np.array([[0,0,0]])  # xyz point of where block should be picked up
        self.waypoints = []
        self.cache = self.camera.BlockFrame
        self.big_contours = {}
        self.small_contours = {}
        self.color_sort_b = {}
        self.color_sort_s = {}
        self.drop_sm = []   #to have a count for number of small blocks placed (see idle function for default positions)
        self.drop_la = []   #to have a count for number of large blocks placed (see idle function for default positions
        # self.target_mask = np.zeros_like(self.camera.DepthFrameRaw, dtype=np.uint8)
        self.targets = []  # list of points ([x,y,z]) for targets
        # [
        #     [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
        #     [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
        #     [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
        #     [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
        #     [0.0,             0.0,      0.0,         0.0,     0.0],
        #     [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
        #     [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
        #     [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
        #     [np.pi/2,         0.5,     0.3,      0.0,     0.0],
        #     [0.0,             0.0,     0.0,      0.0,     0.0]]

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        elif self.next_state == "idle":
            self.idle()

        elif self.next_state == "estop":
            self.estop()

        elif self.next_state == "execute":
            self.execute()

        elif self.next_state == "calibrate":
            self.calibrate()

        elif self.next_state == "calibrate_depth":
            self.calibrate_depth()
        
        elif self.next_state == "calibrate_extrinsic":
            self.calibrate_extrinsic()

        elif self.next_state == "detect":
            self.detect()

        elif self.next_state == "manual":
            self.manual()

        elif self.next_state == "teach_repeat":
            self.teach_repeat()
        
        elif self.next_state == "detect_blocks":
            self.detect_blocks()

        elif self.next_state == "Click_Grab":
            self.Click_Grab()
        
        elif self.next_state == "pick_sort":
            self.Pick_sort()
        
        elif self.next_state == "pick_stack":
            self.pick_stack()
        
        elif self.next_state == "rainbow_sort":
            self.Line_em_up()
        
        elif self.next_state == "rainbow_stack":
            self.stack_em_high()
        
        elif self.next_state == "task_5":
            self.task_5()

    """Functions run for each state"""

    def calibrate(self):
        """!
        @brief      Starting calibration routine. Fist calibrates depth, then extrinsics.
        """
        self.current_state = "calibrate"

        self.next_state = "calibrate_extrinsic"
        self.status_message = "Now calibrating extrinsics using AprilTags. Make sure the AprilTags are visible."

    def calibrate_extrinsic(self):
        """!
        @brief      Performs extrinsic calibration using AprilTag locations
        """
        
        self.current_state = "calibrate_extrinsic"
        rospy.sleep(1)  # sleep to show previously status_message

        ## Tuple of tag poses in world frame (x, y, z)
        model_points = np.array((  
            (-.25, -.025,  0),
            ( .25, -.025,  0), 
            ( .25,  .275,  0),
            (-.25,  .275,  0)#-.05)
        ))
               
        # Perform automatic detection for each individual detected AprilTag
        image_points = np.zeros((4,2))
        for detection in self.camera.tag_detections.detections:
            # print("IDs in detection: ", detection.id)
            # print("Pose of detection: ", detection.pose.pose.pose.position)
            
            tag_pose_cameraframe = np.array([[detection.pose.pose.pose.position.x],
                                             [detection.pose.pose.pose.position.y],
                                             [detection.pose.pose.pose.position.z]])
                                             
            image_point = self.camera.camera2image(tag_pose_cameraframe)
            image_point = np.matmul(np.array([[-1,0],[0,-1]]), image_point - np.array([[640],[360]])) + np.array([[640],[360]])  # Rotate point 180deg about center of image
            image_point = image_point[0:2].reshape(1,2)
            image_points[detection.id[0]-1, :] = image_point
            # print("Tag " + str(detection.id[0]) + " is at mouse point " +  str(image_point) + " and depth " + str(tag_pose_cameraframe[2]))  # detects bottom right of tag
                   
        (_, R_exp, t) = cv2.solvePnP(model_points,
                                     image_points,
                                     self.camera.intrinsic_matrix,
                                     self.camera.distortion,
                                     flags = cv2.SOLVEPNP_ITERATIVE)
        
        R, _ = cv2.Rodrigues(R_exp)
        t *= 1000 
        self.camera.extrinsic_matrix = np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))
        # print("extrinsic=",self.camera.extrinsic_matrix)

        self.next_state = "calibrate_depth"
        self.status_message = "Calibration - calibrating depth"

    def calibrate_depth(self):
        """!
        @brief      Performs extrinsic calibration
        """
        
        self.current_state = "calibrate_depth"
        rospy.sleep(1)  # sleep to show previous status_message

        # ## Tuple of tag poses in world frame (x, y, z)
        # model_points = np.array((  
        #     (-.25, -.025,  0),
        #     ( .25, -.025,  0), 
        #     ( .25,  .275,  0),
        #     (-.25,  .275,  0)#-.05)
        # ))
               
        # # Perform automatic detection for each individual detected AprilTag
        # image_points = np.zeros((4,2))
        # for detection in self.camera.tag_detections.detections:
        #     # print("IDs in detection: ", detection.id)
        #     # print("Pose of detection: ", detection.pose.pose.pose.position)
        #     self.camera.
            
        #     tag_pose_cameraframe = np.array([[detection.pose.pose.pose.position.x],
        #                                      [detection.pose.pose.pose.position.y],
        #                                      [detection.pose.pose.pose.position.z]])
                                             
        #     image_point = self.camera.camera2image(tag_pose_cameraframe)
        #     image_point = np.matmul(np.array([[-1,0],[0,-1]]), image_point - np.array([[640],[360]])) + np.array([[640],[360]])  # Rotate point 180deg about center of image
        #     image_point = image_point[0:2].reshape(1,2)
        #     image_points[detection.id[0]-1, :] = image_point
        #     # print("Tag " + str(detection.id[0]) + " is at mouse point " +  str(image_point) + " and depth " + str(tag_pose_cameraframe[2]))  # detects bottom right of tag
                   
        # (_, R_exp, t) = cv2.solvePnP(model_points,
        #                              image_points,
        #                              self.camera.intrinsic_matrix,
        #                              self.camera.distortion,
        #                              flags = cv2.SOLVEPNP_ITERATIVE)
        
        # R, _ = cv2.Rodrigues(R_exp)
        # t *= 1000 
        # self.camera.extrinsic_matrix = np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))
        
        self.camera.extrinsic_matrix[2,3] += 10
        
        self.camera.cameraCalibrated = True

        self.next_state = "idle"
        self.status_message = "Calibration - Calibration complete."

    def record_waypoint(self):
        positions = list(self.rxarm.get_positions())
        positions.append( self.rxarm.is_gripper_open())
        self.waypoints.append(positions)
        print(self.waypoints)
        self.status_message = "Waypoint " + str(len(self.waypoints)) + " recorded"

    def detect_blocks(self):
        self.current_state = "detect_blocks"

        cv2.destroyAllWindows()        
        #self.camera.BlockFrame = self.cache
        self.camera.detectBlocksInDepthImage()
        #self.camera.processBlockFrame()
        #self.camera.blockDetector()
        self.status_message = "Blocks Detected Boom!"
        self.next_state = "Idle"

    def teach_repeat(self):
        if self.current_state == "estop":  # if current state is estop, don't execute anything
            return
        
        if self.current_state == "idle":
            self.current_state = "teach_repeat"
            self.status_message = "State: Teach and Repeat - Initializing"

            self.rxarm.go_to_sleep_pose(blocking = True)
            rospy.sleep(0.2)
            self.record_waypoint()
            self.rxarm.disable_torque()
            self.rxarm.torque_joints_on(['gripper'])
            rospy.sleep(0.2)
            self.status_message = "State: Teach and Repeat - Initial waypoint recorded"

    def clear_waypoints(self):
        self.waypoints = []
        self.status_message =  "Stored waypoints deleted"  

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

        self.camera.new_click = False
        self.big_contours = {}
        self.small_contours = {}
        self.color_sort_b = {}
        self.color_sort_s = {}
        self.drop_sm = [-400, -125, 10, 0]   #to have a count for number of small blocks placed
        self.drop_la = [400, -125, 20, 0]   #to have a count for number of large blocks placed
        self.targets = []

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        if self.current_state == "estop":  # if current state is estop, don't execute anything
            return
        self.current_state = "execute"
        self.rxarm.enable_torque()
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            if self.current_state == "estop" or self.next_state == "estop":  # if current state is estop, don't execute anything
                return

            arm_displacement = 0
            for j in range(0,5):
                if i>0:
                    arm_displacement += abs(self.waypoints[i][j] - self.waypoints[i-1][j])
            move_time = .375*arm_displacement + 1  # formula for how fast to move arm. Min time = 1 sec
            self.rxarm.set_joint_positions(self.waypoints[i][:5],
                                           moving_time = move_time, 
                                           accel_time = move_time/4,
                                           blocking=True)
            if(self.waypoints[i][5] == True):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()

            rospy.sleep(0.5)
        self.status_message = "State: Execute - Done executing motion plan"
        self.next_state = "idle"
    
    def Click_Grab(self):
        self.status_message = "Click and Grab"

        #dh = pd.read_csv("/home/student/Desktop/Team 107/armlab/config/rx200_dh.csv")
        if self.camera.new_click == True:
            self.camera.new_click = False
            xyz = self.mousexyz_world 

            if self.camera.isSecondClick == False:
                self.block_pickup_xyz = xyz
                xyz[2] -= 20
                self.camera.isSecondClick = True    
                
            else:
                self.camera.isSecondClick = False
                xyz[2] += 30
                phi1,phi2 = self.best_phi(self.block_pickup_xyz, xyz)
                self.moveArm(self.block_pickup_xyz,phi1)        
                self.rxarm.close_gripper()
                rospy.sleep(0.3)
                self.moveArm(xyz, phi2)        
                self.rxarm.open_gripper()        


    ### OTHER FUNCTIONS ####
    def calculate_phi(self,xyz):
        dist = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
        # print("dist", dist)
        # if (dist < 365):
        #     phi = np.pi/2
        # elif dist > 365 and dist < 400:
        #     phi = np.pi/4
        # else:
        #     phi = np.pi/8
        # return phi
        if (dist < 300):
            phi = np.pi/2
        elif dist > 365 and dist < 420:
            phi = np.pi/4
        elif dist > 420 and dist < 440:
            phi = np.pi/8
        else:
            phi = .1
        return phi
    
    def best_phi(self, xyz1, xyz2, phi2=None):
        xyz1 = np.array(xyz1[:3]).reshape((3,1))
        xyz2 = np.array(xyz2[:3]).reshape((3,1))
        phi1 = self.calculate_phi(xyz1)
        if phi2 is None:
            phi2 = self.calculate_phi(xyz2)
        IK12 = kinematics.IK_geometric(self.rxarm.dh_params, np.vstack((xyz1, phi2)))[0] 
        IK21 = kinematics.IK_geometric(self.rxarm.dh_params, np.vstack((xyz2, phi1)))[0] 
        
        if self.rxarm.check_limits(IK12):
            return [phi2,phi2]
        elif self.rxarm.check_limits(IK21):
            return [phi1,phi1]
        else:
            return [phi1, phi2]

    def moveArm(self, xyz, phi=None, theta=None):
        if xyz[2] < 0:
            xyz[2] = 0
        if len(xyz)>3 and phi is None:
            phi = xyz[3]
            xyz = xyz[:3]
        elif phi is None:
            phi = self.calculate_phi(xyz)
        
        xyz = np.reshape(xyz[:3], (3,1))
        pose = np.vstack((xyz, phi))
        target_positions = kinematics.IK_geometric(self.rxarm.dh_params, pose)[0]
        target_positions_offset = kinematics.IK_geometric(self.rxarm.dh_params, pose.flatten() + np.array([0,0,70,0]).flatten())[0]
        if phi > 3*np.pi/8 and theta is not None:
            target_positions[4] = self.rxarm.clamp_wrist_angle(target_positions[0] + theta)
            target_positions_offset[4] = self.rxarm.clamp_wrist_angle(target_positions[0] + theta)

        current_positions = self.rxarm.get_positions()[:5]
        current_pose = kinematics.get_pose_from_T(kinematics.FK_dh(self.rxarm.dh_params, current_positions, 5))
        current_pose_offset = current_pose + np.array([0,0,70,0]).flatten()
        current_positions_offset = kinematics.IK_geometric(self.rxarm.dh_params, current_pose_offset)[0]
        current_positions_offset[4] = current_positions[4]

        # if target_positions is safe, move arm to target, but with 20 degree offset on 
        # joint 2 and 3 (so the arm isn't dragging on build plate)
        if self.isPathSafe(current_pose, pose):
            self.rxarm.set_positions(current_positions_offset, True)                    
            self.rxarm.set_positions(target_positions_offset, True)   
            rospy.sleep(.4)                    
            self.rxarm.set_positions(target_positions, True)
            rospy.sleep(.5)
        else:
            # print('NOT SAFE')
            o = np.pi/180
            self.rxarm.set_positions(current_positions_offset, True) 
            self.rxarm.set_positions(np.array([current_positions[0], -12*o, 38*o, -54*o, 0]), True)                    
            self.rxarm.set_positions(np.array([target_positions[0], -12*o, 38*o, -54*o, 0]), True)       
            self.rxarm.set_positions(target_positions_offset, True)   
            rospy.sleep(.4)              
            self.rxarm.set_positions(target_positions, True)
            rospy.sleep(.5)

    def isPathSafe(self, start_pose, end_pose):
        THRESH_D = 15  # threshold in mm
        THRESH_ANGLE = .2  # threshold in rad
        start_xy = np.array(start_pose[0:2]).flatten()
        end_xy = np.array(end_pose[0:2]).flatten()
        origin_xy = np.array([0,0])
        theta_min = min(np.arctan2(start_xy[1],start_xy[0]), np.arctan2(end_xy[1],end_xy[0]))
        theta_max = max(np.arctan2(start_xy[1],start_xy[0]), np.arctan2(end_xy[1],end_xy[0]))
        d_max = max(np.linalg.norm(start_xy), np.linalg.norm(end_xy))
        for d in self.camera.block_detections:
            block_xy = np.array([d["cx"], d["cy"]]).flatten()
            if np.linalg.norm(block_xy) < d_max+THRESH_D and \
               theta_min - THRESH_ANGLE < np.arctan2(block_xy[1],block_xy[0]) < theta_max + THRESH_ANGLE:
                return False
        return True

    def sort_rainbow(self):
        self.color_sort_b = {}
        self.color_sort_s = {}
        self.color_sort_b = sorted(self.big_contours, key= lambda d: d["rank"])
        self.color_sort_s = sorted(self.small_contours, key= lambda d: d["rank"])

    def isTarget(self, detection):
        for target_xyz in self.targets:
            print([detection["cx"], detection["cy"], detection["z"]])
            detection_xyz = np.array([detection["cx"],detection["cy"],detection["z"]]).flatten()            
            if np.linalg.norm(target_xyz.flatten() - detection_xyz) < 100:
                return True
        return False

    # def mask(self, xyz, width, height):        
        # cv2.rectangle(self.target_mask, (xyz[0]-width/2,xyz[1]-height/2),((xyz[0]+width/2,xyz[1]+height/2), 255, cv2.FILLED))
        # cv2.rectangle(self.target_mask, (xyz[0]-width/2,xyz[1]-height/2),((xyz[0]+width/2,xyz[1]+height/2), (255, 0, 0), 2))

    def generate_blocks_size_wise(self):
        self.big_contours = []
        self.small_contours = []
        for detection in self.camera.block_detections:
            if not self.isTarget(detection) and detection["cy"] >0:
                if 750 < detection["area"] < 1800:
                    self.big_contours.append(detection)
                elif 200 < detection["area"] <= 750:
                    self.small_contours.append(detection)
                else:
                    print("ERROR: block not within size bounds. Found area " + str(detection["area"]) + " at (" + str(detection["cx"]) + ", " + str(detection["cy"]) + ")")
          
    def generate_pick_pose(self,detection,is_small):
        phi = self.calculate_phi([detection["cx"], detection["cy"], detection["z"]])
        pick_pose = [detection["cx"], detection["cy"], detection["z"],phi]
        if is_small:
            pick_pose[2] -= 8  # account for smaller blocks
        else:
            pick_pose[2] -= 15
        return pick_pose 
    
    def num_detections_in_positive_plane(self):
        count = 0
        for detection in self.camera.block_detections:
            if detection["cy"] > 10:
                count += 1
                print("detection['cx'],detection['cy']",detection["cx"],detection["cy"])
        return count

    def sort(self, detections, type_detection, stack_height=None):        
        if type_detection == "large":
            for detection in detections:
                if not self.isTarget(detection) and detection["cy"]>10:
                    pick_pose = self.generate_pick_pose(detection, False)    
                    phi1,phi2 = self.best_phi(pick_pose, self.drop_la)
                    self.moveArm(pick_pose,phi=phi1, theta=detection["theta"])
                    self.rxarm.close_gripper()
                    print("self.get_gripper_position()", self.rxarm.get_gripper_position())
                    if self.rxarm.get_gripper_position() < 0.035:
                        self.rxarm.open_gripper()
                        self.sort([detection], type_detection="small", stack_height=stack_height)
                        rospy.sleep(.3)                
                    else:
                        self.moveArm(self.drop_la, phi=phi2, theta = np.pi/2)
                        self.rxarm.open_gripper()
                        self.targets.append(np.array(self.drop_la[:3]).flatten())
                        if self.drop_la[0] >= 150 and stack_height is not None and self.drop_la[2] < (stack_height-1)*42:
                            self.drop_la[2] += 42
                        elif self.drop_la[0] >= 150:
                            self.drop_la[2] = 20
                            self.drop_la[0] -= 100
                            if stack_height is not None:
                                self.drop_la[0] -= 50  # extra buffer to avoid stacks
                        else:
                            self.drop_la[2] = 20
                            self.drop_la[0] = 365
                            self.drop_la[1] += 75
                            

        elif type_detection == "small":
            for detection in detections:
                if not self.isTarget(detection) and detection["cy"]>10:                    
                    pick_pose = self.generate_pick_pose(detection, True)                    
                    # print("z", self.drop_sm[2])
                    phi1,phi2 = self.best_phi(pick_pose, self.drop_sm)
                    self.moveArm(pick_pose,phi=phi1, theta=detection["theta"])
                    self.rxarm.close_gripper()
                    print("self.get_gripper_position()", self.rxarm.get_gripper_position())
                    if self.rxarm.get_gripper_position() > 0.035:
                        self.rxarm.open_gripper()
                        self.sort([detection], type_detection="large", stack_height=stack_height)
                        rospy.sleep(.3)                     
                    else:
                        self.moveArm(self.drop_sm,phi=phi2, theta = np.pi/2)
                        self.rxarm.open_gripper()
                        # self.targets.append(np.array(self.drop_sm[:3]).flatten())
                        # print("self.drop_la[2]", self.drop_la[2])
                        # print("stack_height", stack_height)
                        if self.drop_sm[0] <= -150 and stack_height is not None and self.drop_sm[2] < (stack_height-1)*28:
                            self.drop_sm[2] += 28                    
                        elif self.drop_sm[0] <= -150:
                            self.drop_sm[2] = 15
                            self.drop_sm[0] += 55   
                            if stack_height is not None:
                                self.drop_sm[0] += 50  # extra buffer to avoid stacks     
                        else:
                            self.drop_sm[2] = 15
                            self.drop_sm[0] = -365
                            self.drop_sm[1] += 75
                            
        else:
            print("UNKNOWN TYPE")

    # Task 1
    def Pick_sort(self):
        self.detect_blocks()
        self.generate_blocks_size_wise()
        # while self.num_detections_in_positive_plane()>0:
        print("len(self.big_contours)", len(self.big_contours))
        print("len(self.small_contours)",len(self.small_contours))
        while len(self.big_contours)>0 or len(self.small_contours)>0:
            self.rxarm.initialize()
            self.sort(self.big_contours, type_detection="large")                        
            self.sort(self.small_contours, type_detection="small")                      
            self.rxarm.sleep()          
            self.detect_blocks()   
            self.generate_blocks_size_wise()
        print("Finished Task 1")
        self.next_state = "idle"            

    # Task 2
    def pick_stack(self):
        self.drop_sm = [-300, -125, 20, 0]   #to have a count for number of small blocks placed
        self.drop_la = [300, -125, 25, 0]   #to have a count for number of large blocks placed
        
        self.detect_blocks()
        self.generate_blocks_size_wise()
        # while self.num_detections_in_positive_plane()>0:
        while len(self.big_contours)>0 or len(self.small_contours)>0:            
            self.rxarm.initialize()
            self.sort(self.big_contours, "large", stack_height=3)                        
            self.sort(self.small_contours, "small", stack_height=3)                      
            self.rxarm.sleep()      
            self.detect_blocks() 
            self.generate_blocks_size_wise() 
        print("Finished Task 2")
        self.next_state = "idle"   

    def position_from_rank(self, rank):
        xy = {
            1: [350, -125],
            2: [350, -75],
            3: [350, -25],
            4: [350, 25],
            5: [350, 75],
            6: [350, 125],
            7: [350, 175]
        }
        return xy.get(rank)
    # Task 3
    def sort_colors(self,detections, size):
        pass
        # if size == "large":
        #     for detection in detections:
        #         if not self.isTarget(detection):
        #             pick_pose = self.generate_pick_pose(detection, False)   
        #             #self.drop_la                 
        #             drop_z_prev = self.drop_la[2]
        #             self.drop_la[2] += (current_stack_height-1)*45
        #             #self.drop_la[3] = self.calculate_phi(self.drop_la[:3])
        #             phi1,phi2 = self.best_phi(pick_pose, self.drop_la)
        #             self.moveArm(pick_pose,phi=phi1, theta=detection["theta"])
        #             self.rxarm.close_gripper()
        #             print("self.get_gripper_position()", self.rxarm.get_gripper_position())
        #             if self.rxarm.get_gripper_position() < 0.04:
        #                 self.sort(detection, type_detection="small")
        #                 rospy.sleep(.3)                
                    
        #             else:
        #                 self.moveArm(self.drop_la, phi=phi2, theta = 0)
        #                 self.rxarm.open_gripper()
        #                 self.targets.append(np.array(self.drop_la[:3]).flatten())
        #                 if self.drop_la[0] >= 150 and stack_height is not None and current_stack_height < stack_height:
        #                     current_stack_height += 1
        #                 elif self.drop_la[0] >= 150:
        #                     current_stack_height = 1
        #                     self.drop_la[0] -= 75
        #                 else:
        #                     current_stack_height = 1
        #                     self.drop_la[0] = 365
        #                     self.drop_la[1] += 75
        #                     if stack_height is not None:
        #                         self.drop_la[1] += 60  # extra buffer to avoid stacks
        #                 self.drop_la[2] = drop_z_prev





    def Line_em_up(self):  
        self.detect_blocks()
        self.generate_blocks_size_wise()
        while len(self.big_contours)>0 or len(self.small_contours)>0:
            self.rxarm.initialize()
            self.sort_rainbow()
            self.sort(self.color_sort_b, "large")
            self.sort(self.color_sort_s, "small")                      
            self.rxarm.sleep()      
            self.detect_blocks()   

        print("Finished Task 3")
        self.next_state = "idle"  
                                    
    def stack_em_high(self):
        drop_sm = [-400, -250, -3, 0]   
        drop_la = [400, -250, -5, 0]  
        i_la = 0    #to have a count for number of small blocks placed
        i_sm = 0   #to have a count for number of large blocks placed
        while True:
            self.detect_blocks()
            num_of_detections = len(self.camera.block_detections)
            if num_of_detections:
                self.generate_blocks_size_wise()
                self.sort_rainbow()
                for detection in self.color_sort_b:
                    pick_pose = self.generate_pick_pose(detection)
                    self.moveArm(pick_pose)
                    self.rxarm.close_gripper()
                    if i_la <3:
                        drop_la[2] = i_la*40
                        drop_la[3] = self.calculate_phi(drop_la[:3])
                        self.moveArm(drop_la)
                        self.rxarm.open_gripper()
                        self.mask(drop_la, 40,40)
                        i_la+=1
                    else:
                        drop_la[0] -= 75
                        drop_la[1] += 100
                        drop_la[2] = 0
                        self.moveArm(drop_la)
                        self.rxarm.open_gripper()
                        self.mask(drop_la,40,40)
                        i_la = 1
                for detection in self.color_sort_s:
                    pick_pose = self.generate_pick_pose(detection)
                    self.moveArm(pick_pose)
                    self.rxarm.close_gripper()
                    if i_sm <3:
                        drop_sm[2] = i_sm*30
                        drop_sm[3] = self.calculate_phi(drop_sm[:3])
                        self.moveArm(drop_sm)
                        self.rxarm.open_gripper()
                        self.mask(drop_la, 35,35) 
                        
                        i_sm+=1
                    else:
                        drop_sm[0]+=35
                        drop_sm[1] += 100
                        drop_sm[2] = 0
                        self.moveArm(drop_sm)
                        self.rxarm.open_gripper()
                        self.mask(drop_la, 35,35) 
                        i_sm = 1            

    def task_5(self):
        self.drop_sm = [-400, -150, -3, 0]   
        self.drop_la = [250, 0, 15, 0]
        i_la = 0    #to have a count for number of small blocks placed
        self.detect_blocks()
        self.generate_blocks_size_wise()
        # while self.num_detections_in_positive_plane()>0:
        while len(self.big_contours)>0:
            self.rxarm.initialize()

            # if self.small_contours:
            #     self.sort(self.small_contours, "small")
                                
            for detection in self.big_contours: 
                if not (self.drop_la[0] - 100 < detection["cx"] < self.drop_la[0] + 100 and \
                        self.drop_la[1] - 100 < detection["cy"] < self.drop_la[1] + 100):                              
                    pick_pose = self.generate_pick_pose(detection, False) 
                    phi2 = np.pi/2
                    if (self.drop_la[2] < 160):
                        phi2 = np.pi/2
                    elif self.drop_la[2] > 160 and self.drop_la[2] < 250:
                        phi2 = np.pi/4
                    elif self.drop_la[2] > 250 and self.drop_la[2] < 420:
                        phi2 = 0
                    else:
                        phi2 = -np.pi/4
                    phi1,phi2 = self.best_phi(pick_pose, self.drop_la)
                    self.moveArm(pick_pose,phi=phi1, theta=detection["theta"])
                    self.rxarm.close_gripper()
                    #print("self.get_gripper_position()", self.rxarm.get_gripper_position())
                    # if self.rxarm.get_gripper_position() < 0.035:
                    #     self.rxarm.open_gripper()
                    #     self.sort([detection], type_detection="small")
                    #     rospy.sleep(.3)                
                    # else:
                    self.moveArm(self.drop_la, phi=phi2, theta = np.pi/2)
                    self.rxarm.open_gripper()
                    self.targets.append(np.array(self.drop_la[:3]).flatten())
                    self.drop_la[2] += 42
                else:
                    print("Target avoided")

            self.rxarm.sleep()      
            self.detect_blocks() 
            self.generate_blocks_size_wise() 
        
        print("Finished Task 5")
        self.next_state = "idle" 
                      
                # pick_pose = self.generate_pick_pose(detection)
                # self.moveArm(pick_pose)
                # self.rxarm.close_gripper()
                # if i_la <3:
                #     drop_la[2] = i_la*40
                #     drop_la[3] = self.calculate_phi(drop_la[:3])
                #     self.moveArm(drop_la)
                #     self.rxarm.open_gripper()
                #     self.mask(drop_la, 40,40)
                #     i_la+=1
                # else:
                #     drop_la[0] -= 75
                #     drop_la[1] += 100
                #     drop_la[2] = 0
                #     self.moveArm(drop_la)
                #     self.rxarm.open_gripper()
                #     self.mask(drop_la,40,40)
                #     i_la = 1
                    

    """ TODO """
    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)