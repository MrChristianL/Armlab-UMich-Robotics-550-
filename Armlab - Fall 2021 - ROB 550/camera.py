"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.BlockFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.moments = np.array([[]])
        # self.colors = list((
        #                       {'id': 'red', 'lower': [204,204,255], 'upper': [0,0,51]},
        #                       {'id': 'orange', 'lower': [204,229,255], 'upper': [0,25,51]},
        #                       {'id': 'yellow', 'lower': [204,255,255], 'upper': [0,51,51]},
        #                       {'id': 'green', 'lower': [204,255,204]},
        #                       {'id': 'blue', 'lower': 110},
        #                       {'id': 'violet', 'lower': 133},
        #                     )
        #                   )
        # self.colors = list((
        #     {'id': 'Red', 'color': (10, 10, 127)},
        #     {'id': 'Orange', 'color': (30, 75, 150)},  #sometimes detecting as yellow
        #     {'id': 'Yellow', 'color': (30, 150, 200)},
        #     {'id': 'Green', 'color': (20, 60, 20)},
        #     {'id': 'Blue', 'color': (100, 50, 0)},  
        #     {'id': 'Violet', 'color': (100, 40, 80)}) # always detecting as blue
        # )
        self.colors = list((
            {'id': 'red', 'color': 123},
            {'id': 'orange', 'color': 110},
            {'id': 'yellow', 'color': 95},
            {'id': 'green', 'color': 46},
            {'id': 'blue', 'color': 15},
            {'id': 'violet', 'color': 3}
        ))

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        # self.intrinsic_matrix = np.array([[965.123004, 0, 648.6029343],[0, 964.4260055, 374.690494],[0, 0, 1]])  # from our checkboard calibration
        # self.intrinsic_matrix = np.array([[918.3599853515625, 0.0, 661.1923217773438], [0.0, 919.1538696289062, 356.59722900390625], [0.0, 0.0, 1.0]])  # from factory settings
        # self.intrinsic_matrix = np.array([[904.31762695, 0, 644.01403809], [0, 904.82452393, 360.77752686], [0, 0, 1]])  # from current camera settings
        self.intrinsic_matrix = np.array([])  # will be updated by ROS message cameraInfo
        # self.extrinsic_matrix = np.array([[1, 0, 0, 0],[0,-1,0,175],[0,0,-1,968.375],[0,0,0,1]])  # from our naive calculations
        self.extrinsic_matrix = np.array([[1, 0, 0, 35],[0,-1,0,120],[0,0,-1,976.375],[0,0,0,1]])  # after tuning
        # self.extrinsic_matrix = np.matrix([[1., 0, 0, -15], [0, -1., 0, 220.], [0, 0, -1., 985.], [0, 0, 0, 1.]])  # from professor
        # self.distortion = np.array([0.14005725,	-0.231303,	0.002595,	-0.00038125])
        self.distortion = np.array([ 0.15564486384391785, -0.48568257689476013, -0.0019681642297655344, 0.0007267732871696353, 0.44230175018310547])  #from factory settings

        self.last_click = np.array([0, 0])
        self.new_click = False
        self.isSecondClick = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processBlockFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.BlockFrame, self.block_contours, -1,
                         (0, 255, 255), thickness=1)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw #>> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadBlockFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.BlockFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None
    
    def convertQtBlockFrame(self):
        """!
        @brief      Converts Block frame to format suitable for Qt

        @return     QImage
        """
        try:
            frame = cv2.resize(self.BlockFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

        @param      file  The file
        """
        # for i in range(21):
        #     if i <9:
        #         self.intrinsic_matrix = np.array(file.read().readlines())
        #     else:
        #         self.extrinsic_matrix = np.array(file.read().readlines())
        # self.intrinsic_matrix.reshape(3,3)
        # self.extrinsic_matrix.reshape(3,4)
        # TODO: push intrinsic and extrinsic to camera
        pass
    
    def transform_matrix_inv(self, H):
        n = H.shape[0]-1
        H[0:n,0:n] = np.transpose(H[0:n,0:n]) 
        H[0:n,n] = -np.matmul(H[0:n,0:n],H[0:n,n])
        return H

    def Pixel_world(self, p):
        pt = np.vstack((p.x(), p.y(), 1))
        inverse = np.linalg.inv(self.intrinsic_matrix)
        camera_coordinates_point1 = np.matmul(inverse, pt)
        #print(camera_coordinates_point1)
        camera_coordinates_point = np.append(camera_coordinates_point1, [[1]], axis=0)
        
        #print(camera_coordinates_point, camera_coordinates_point.shape) 
        # print(camera_coordinates_point[0],camera_coordinates_point[1])       
        # world_coordinates = np.matmul(self.extrinsic_matrix, camera_coordinates_point)
        # print("extrinsic matrix", self.extrinsic_matrix)
        # print("world coordinates", world_coordinates)
        return camera_coordinates_point1

    def camera2image(self, xyz_cam):
        """!
        @brief      Converts camera frame to image plane using instrinsic matrix

        @param      xyz_cam vector of x,y,z coordinates in the camera frame (size 3 vector)

        @return     vector of u,v in image plane (size 2 vector)
        """
        z_c = xyz_cam[2]
        p_i = 1/z_c * np.matmul(self.intrinsic_matrix, xyz_cam.reshape([3,1]))
        return p_i[0:2].reshape(2,1)

    def world2camera(self, xyz_world):
        """!
        @brief      Converts world frame to camera frame using extrinsic matrix

        @param      xyz_world vector of x,y,z coordinates in the world frame (size 3 vector)

        @return     vector of x,y,z in the camera frame (size 3 vector)
        """
        p_world = np.row_stack((xyz_world.reshape([3,1]), [1]))
        p_cam = np.matmul(self.extrinsic_matrix, p_world)
        return p_cam[0:3] 

    def image2world(self, uvd): 
        """!
        @brief      Converts image plane (plus depth) to world frame 

        @param      uvd: vector of u,v,d, where [u,v] are image coordinates and d is depth at that point (size 3 vector)

        @return     vector of x,y,z in the world frame (size 3 vector)
        """
        p_image = np.row_stack((uvd[0:2].reshape([2,1]), [1]))  # [u; v; 1]
        K_inv = np.linalg.inv(self.intrinsic_matrix)
        xyz_cam = uvd[2] * np.matmul(K_inv, p_image)
        p_cam = np.row_stack((xyz_cam, [1])) # [x_c; y_c; z_c; 1]
        
        H_inv = np.linalg.inv(self.extrinsic_matrix)
        p_world = np.matmul(H_inv, p_cam)  # [x_w; y_w; z_w; 1]
        return p_world[0:3] 


    def Colour_detector(self, contour, cx, cy):
        
        hsv = cv2.cvtColor(self.VideoFrame, cv2.COLOR_BGR2HSV)
        h_surrounding = hsv[cy-2:cy+2,cx-2:cx+2,0]
        
        h = float(cv2.mean(h_surrounding)[0])
        
        hsv_ = hsv[cy,cx,:]
        min_dist = (np.inf, None)
        for label in self.colors:
            d = abs(label["color"] - h)
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        return min_dist[1] 

    def shape_detector(self, contour):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
        if len(approx) <= 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            a = w / float(h)
            if 0.70 <= a <= 1.40:
               return True
            else:
                return False
        else:
            return False

    def rank_color(self, color):
        switcher = {
           "Violet" : 1,
           "Indigo" : 2,
           "Blue"   : 3,
           "Green"  : 4,
           "Yellow" : 5,
           "Orange" : 6,
           "Red"    : 7
         }
        return switcher.get(color)
        

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        for contour in self.block_contours:
            max_area = 5000.00 #in mm
            min_area = 450.00 #in mm
            
            if min_area < cv2.contourArea(contour) < max_area:
                cv2.drawContours(self.BlockFrame, contour, -1,(255, 255, 255), thickness=2)
            if self.shape_detector(contour) == True:
                cv2.drawContours(self.BlockFrame, contour, -1,(255, 0, 0), thickness=4)
                
            if min_area < cv2.contourArea(contour) < max_area and self.shape_detector(contour) == True:
                #cv2.drawContours(self.BlockFrame, contour, -1,(0, 255, 255), thickness=2)
                min_area_rect = cv2.minAreaRect(contour)
                theta = float(min_area_rect[2])
                (width, height) = min_area_rect[1]
                width, height = float(width), float(height)
                area = float(width*height)
                
                M = cv2.moments(contour)
                cx_ = int(M['m10']/M['m00'])
                cy_ = int(M['m01']/M['m00'])
                z_ = self.DepthFrameRaw[cy_][cx_]
                (cx,cy,z) = self.image2world(np.array([cx_,cy_,z_]))
                cx,cy,z = float(cx), float(cy), float(z)

                # perimeter = cv2.arcLength(contour, True)
                # approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                # (x, y, w, h) = cv2.boundingRect(approx)
                # area = float(w*h)

                isDuplicate = False
                for i in range(len(self.block_detections)):
                    d = self.block_detections[i]
                    cx_d,cy_d = d["cx"],d["cy"]
                    if abs(cx_d-cx) < 15 and abs(cy_d-cy) < 15:
                        isDuplicate = True
                        # if it's a duplicate, take the bigger area
                        if d["area"]> area:
                            color = self.Colour_detector(contour,cx_,cy_)
                            rank = self.rank_color(color)
                            self.block_detections[i] = \
                                {"cx":cx, "cy":cy, "z":z, "width":width, "height":height, "area":area, 
                                "theta":theta, "color":color, "contour":contour, "rank": rank}
                        
                            cv2.putText(self.BlockFrame, color, (cx_-25, cy_+40), self.font, 0.5, (255,255,255), thickness=1)
                            cv2.putText(self.BlockFrame, "theta=" + str(int(theta)), (cx_-40, cy_+60), self.font, 0.5, (255,255,255), thickness=1)
                            cv2.putText(self.BlockFrame, "area=" + str(int(area)), (cx_-40, cy_+80), self.font, 0.5, (255,255,255), thickness=1)
                            box = np.int0(cv2.boxPoints(min_area_rect))
                            cv2.drawContours(self.BlockFrame, [box], 0, (36,255,12), 1) # OR


                
                if not isDuplicate:
                    color = self.Colour_detector(contour,cx_,cy_)
                    rank = self.rank_color(color)
                    self.block_detections.append(
                        {"cx":cx, "cy":cy, "z":z, "width":width, "height":height, "area":area, 
                        "theta":theta, "color":color, "contour":contour, "rank": rank})
                
                    cv2.putText(self.BlockFrame, color, (cx_-25, cy_+40), self.font, 0.5, (255,255,255), thickness=1)
                    cv2.putText(self.BlockFrame, "theta=" + str(int(theta)), (cx_-40, cy_+60), self.font, 0.5, (255,255,255), thickness=1)
                    cv2.putText(self.BlockFrame, "area=" + str(int(area)), (cx_-40, cy_+80), self.font, 0.5, (255,255,255), thickness=1)
                    box = np.int0(cv2.boxPoints(min_area_rect))
                    cv2.drawContours(self.BlockFrame, [box], 0, (36,255,12), 1) # OR
                
            
            #print(color, int(theta), cx, cy)


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        self.block_detections = []
        self.BlockFrame = np.zeros_like(self.BlockFrame)
        
        lower = 505
        upper = 960
        for i in range(lower,upper,5):
            mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
            # TODO COMPETITION mask out target points as well
            cv2.rectangle(mask, (231,56),(1130,613), 255, cv2.FILLED)
            cv2.rectangle(mask, (582,317),(779,613), 0, cv2.FILLED)
            cv2.rectangle(self.BlockFrame, (231,56),(1130,613), (255, 0, 0), 2)
            cv2.rectangle(self.BlockFrame, (582,317),(779,613), (255, 0, 0), 2)
            
            thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, i-10, i+10), mask)
            #print("lower and upper", i, i+10)
            #thresh = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)
            _, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.block_contours = contours
            self.blockDetector()
        
        for d in self.block_detections:
            # print("d=",d)
            cv2.drawContours(self.BlockFrame,d["contour"], -1, (0, 255, 255), thickness=2)
            # cv2.putText(self.BlockFrame, d["color"], (d["cx"]-30, d["cy"]+40), self.font, 0.5, (255,255,255), thickness=1)
            # cv2.putText(self.BlockFrame, str(int(d["theta"])), (d["cx"], d["cy"]), self.font, 0.5, (255,255,255), thickness=1)
        

class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image




class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
    
    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blocks Window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_frame = self.camera.convertQtBlockFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    "Blocks Window",
                    cv2.cvtColor(self.camera.BlockFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
