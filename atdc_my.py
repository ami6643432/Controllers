#!/usr/bin/env python
import sys
# ROS python API
import rospy

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, AccelStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import *
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from msg_check.msg import PlotDataMsgTDC

from scipy import linalg as la


import numpy as np
from tf.transformations import *
#import RPi.GPIO as GPIO
import time

# Kpos = np.array([-2, -2, -3])
# Kvel = np.array([-2, -2, -3])
# Flight modes class
# Flight modes are activated using ROS services
class fcuModes:
    def __init__(self):
        pass

    def setTakeoff(self):
        rospy.wait_for_service('mavros/cmd/takeoff')
        try:
            takeoffService = rospy.ServiceProxy('mavros/cmd/takeoff', mavros_msgs.srv.CommandTOL)
            takeoffService(altitude = 3)
        except rospy.ServiceException as e:
            print ("Service takeoff call failed: %s",e)

    def setArm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(True)
        except rospy.ServiceException as e:
            print ("Service arming call failed: %s",e)

    def setDisarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            armService(False)
        except rospy.ServiceException as e:
            print ("Service disarming call failed: %s",e)

    def setStabilizedMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='STABILIZED')
        except rospy.ServiceException as e:
            print ("service set_mode call failed: %s. Stabilized Mode could not be set.",e)

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print ("service set_mode call failed: %s. Offboard Mode could not be set.",e)

    def setAltitudeMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='ALTCTL')
        except rospy.ServiceException as e:
            print ("service set_mode call failed: %s. Altitude Mode could not be set.",e)

    def setPositionMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='POSCTL')
        except rospy.ServiceException as e:
            print ("service set_mode call failed: %s. Position Mode could not be set.",e)

    def setAutoLandMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
            flightModeService(custom_mode='AUTO.LAND')
        except rospy.ServiceException as e:
               print ("service set_mode call failed: %s. Autoland Mode could not be set.",e)

class Controller:
    # initialization method
    def __init__(self):
        # Drone state
        self.state = State()
        # Instantiate a setpoints message
        self.sp = PoseStamped()
        # set the flag to use position setpoints and yaw angle
       
        # Step size for position update
        self.STEP_SIZE = 2.0
        # Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

        # A Message for the current local position of the drone

        # initial values for setpoints
        self.cur_pose = PoseStamped()
        self.cur_vel = TwistStamped()
        self.cur_acc = AccelStamped()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.ALT_SP = 0.5
        self.sp.pose.position.z = self.ALT_SP
        self.local_pos = Point(0.0, 0.0, self.ALT_SP)
        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.errInt = np.zeros(3)
        self.att_cmd = PoseStamped()
        self.thrust_cmd = Thrust()

        # Control parameters

        #TDC

        self.M_bar = 1.2 * np.identity(3)
        # self.M_bar = np.array([[3, 0, 1],[0, 3, 1],[1, 1, 1]])

        self.a = np.array([0.001,0.001,1])
        
        self.alpha = 1

        self.beta0_cap = 0.01

        self.beta1_cap = 0.01

        self.gamma = 5

        self.epsilon = 0.1

        self.Kp = np.array([[3, 0, 0],[0, 3, 0],[0, 0, 3.5]])
   
        self.Kd = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 4]])

        
   


        # self.N_cap = np.array([1.0, 1.0, 1.0])

        self.pre_th = np.array([0, 0, 0])

        # self.pre_pose = np.array([0, 0, 0])

        self.pre_pose = np.array([self.cur_pose.pose.position.x , self.cur_pose.pose.position.y, self.cur_pose.pose.position.z])

        self.pre_acc = np.array([self.cur_acc.accel.linear.x , self.cur_acc.accel.linear.y, self.cur_acc.accel.linear.z])

        self.s_prev = 0
        


        ##ASMC
        # self.Kp0 = np.array([1.0, 1.0, 1.0])
        # self.Kp1 = np.array([2.0, 2.0, 1.0])
        # self.Lam = np.array([3.0, 3.0, 4.0])
        # self.Phi = np.array([1.5, 1.5, 1.0])
        # self.M = 0.1
        # self.alpha_0 = np.array([1,1,1])
        # self.alpha_1 = np.array([3,3,3])
        # self.alpha_m = 0.02

        self.v = 0.1

        self.norm_thrust_const = 0.06
        self.max_th = 12.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])
        self.pre_time = rospy.get_time()    
        self.data_out = PlotDataMsgTDC()

        # Publishers
        self.att_pub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
        self.thrust_pub = rospy.Publisher('mavros/setpoint_attitude/thrust', Thrust, queue_size=10)
        self.data_pub = rospy.Publisher('/data_out', PlotDataMsgTDC, queue_size=10)
        self.armed = False
        self.pin_1 = 16
        self.pin_2 = 18
        # GPIO.setmode(GPIO.BOARD)
        # GPIO.setup(self.pin_1, GPIO.OUT)
        # GPIO.setup(self.pin_2, GPIO.OUT)


        # speed of the drone is set using MPC_XY_CRUISE parameter in MAVLink
        # using QGroundControl. By default it is 5 m/s.

    # Callbacks



	# def multiDoFCb(self, msg):

    def multiDoFCb(self, msg):
        pt = msg.points[0]
        self.sp.pose.position.x = pt.transforms[0].translation.x
        self.sp.pose.position.y = pt.transforms[0].translation.y
        self.sp.pose.position.z = pt.transforms[0].translation.z
        
        #self.data_out.posn = self.sp.pose.position
        
        self.desVel = np.array([pt.velocities[0].linear.x, pt.velocities[0].linear.y, pt.velocities[0].linear.z])

        self.desAcc = np.array([pt.accelerations[0].linear.x, pt.accelerations[0].linear.y, pt.accelerations[0].linear.z])
        
        # self.array2Vector3(sp.pose.position, self.data_out.acceleration)

        if (pt.transforms[0].translation.x < 0.01 and pt.transforms[0].translation.x > -0.01) \
        and (pt.transforms[0].translation.y > -1.01 and pt.transforms[0].translation.y < -0.99 ):            
            # GPIO.output(self.pin_1, True)
            print("Dropping first payload")
            time.sleep(0.1)
            # self.switch = 2.0
            # print(self.switch)
        
        elif (pt.transforms[0].translation.x < 0.01 and pt.transforms[0].translation.x > -0.01) \
        and (pt.transforms[0].translation.y < 1.01 and pt.transforms[0].translation.y > 0.99 ):
            # GPIO.output(self.pin_2, True)
            print("Dropping second payload")
            time.sleep(0.1)
            # self.switch = 1.0
            # print(self.switch)





    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
        self.local_quat[0] = msg.pose.orientation.x
        self.local_quat[1] = msg.pose.orientation.y
        self.local_quat[2] = msg.pose.orientation.z
        self.local_quat[3] = msg.pose.orientation.w

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    ## Update setpoint message
    def updateSp(self):
        self.sp.pose.position.x = self.local_pos.x
        self.sp.pose.position.y = self.local_pos.y
        # self.sp.position.z = self.local_pos.z

    def odomCb(self, msg):
        self.cur_pose.pose.position.x = msg.pose.pose.position.x
        self.cur_pose.pose.position.y = msg.pose.pose.position.y
        self.cur_pose.pose.position.z = msg.pose.pose.position.z

        self.cur_pose.pose.orientation.w = msg.pose.pose.orientation.w
        self.cur_pose.pose.orientation.x = msg.pose.pose.orientation.x
        self.cur_pose.pose.orientation.y = msg.pose.pose.orientation.y
        self.cur_pose.pose.orientation.z = msg.pose.pose.orientation.z

        self.cur_vel.twist.linear.x = msg.twist.twist.linear.x
        self.cur_vel.twist.linear.y = msg.twist.twist.linear.y
        self.cur_vel.twist.linear.z = msg.twist.twist.linear.z

        self.cur_vel.twist.angular.x = msg.twist.twist.angular.x
        self.cur_vel.twist.angular.y = msg.twist.twist.angular.y
        self.cur_vel.twist.angular.z = msg.twist.twist.angular.z

    def newPoseCB(self, msg):
        if(self.sp.pose.position != msg.pose.position):
            print("New pose received")
        self.sp.pose.position.x = msg.pose.position.x
        self.sp.pose.position.y = msg.pose.position.y
        self.sp.pose.position.z = msg.pose.position.z
   
        self.sp.pose.orientation.x = msg.pose.orientation.x
        self.sp.pose.orientation.y = msg.pose.orientation.y
        self.sp.pose.orientation.z = msg.pose.orientation.z
        self.sp.pose.orientation.w = msg.pose.orientation.w

    def vector2Arrays(self, vector):        
        return np.array([vector.x, vector.y, vector.z])

    def array2Vector3(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]

    # Defining sigmoid function for adaptive part

    def sigmoid(self, s, e):
        sig = s/np.sqrt( np.square(np.linalg.norm(s)) + e )
        return sig


    def th_des(self):
        dt = rospy.get_time() - self.pre_time
        self.pre_time = self.pre_time + dt
        if dt > 0.04:
            dt = 0.04

        curPos = self.vector2Arrays(self.cur_pose.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.cur_vel.twist.linear)
        curAccLin = self.vector2Arrays(self.cur_acc.accel.linear)
        curAccAng = self.vector2Arrays(self.cur_acc.accel.angular)

        errPos = desPos - curPos
        print("Err : ", errPos)
        errVel = self.desVel - curVel

        # print("Position : ",curPos)

        E = np.transpose(np.hstack((errPos,errVel)))
        # print("E :", E)

        A_top = np.hstack((np.zeros([3,3]),np.identity(3)))
        A_bot = np.hstack((-self.Kp,-self.Kd))
        A = np.vstack((A_top,A_bot))

        Q = np.identity(6)

        P = la.solve_continuous_lyapunov(A,Q)

        B = np.vstack((np.zeros([3,3]),np.identity(3)))
        # print("B :", B)
        # print("P :", P)
        # print("E :", E)

        s = (np.transpose(B)) @ P @ E
        # print("s :", s)

        # Calculations

        self.beta0_prev = self.beta0_cap
        self.beta1_prev = self.beta1_cap

        beta0_prev = self.beta0_prev            
        beta1_prev = self.beta1_prev  

        # self.s_prev =   self.Kp*self.errPosPrev + self.Kd*self.errVelPrev

        beta0_dot = 0
        beta1_dot = 0

        s_norm = np.linalg.norm(s)
        s_prev_norm = np.linalg.norm(self.s_prev)
        E_norm = np.linalg.norm(E)

        # Updating beta
        
        if s_norm > s_prev_norm: 
            beta0_dot = self.gamma*s_norm
            beta1_dot = self.gamma*E_norm*s_norm

        elif s_norm <= s_prev_norm:
            beta0_dot = -self.gamma*s_norm
            beta1_dot = -self.gamma*E_norm*s_norm

    
        beta0 = beta0_prev + beta0_dot*dt
        beta1 = beta1_prev + beta1_dot*dt

        self.c = beta0 + beta1 * E_norm

        self.adapt = self.alpha * self.c * self.sigmoid(s , self.epsilon)

        # self.du = (self.c)*(s/np.sqrt(np.square(s_norm)+np.square(self.epsilon)))
        
        # self.u = self.desAcc - self.Kd*errVel - self.Kp*errPos + self.alpha*self.du


        
        u0 = self.desAcc + self.Kd @ errVel + self.Kp @ errPos
        
        # print("u0 :", u0)

        self.N_cap = self.pre_th - self.M_bar @ self.pre_acc

        # print("N_cap :", self.N_cap)

        self.N_cap = self.N_cap * self.a

        # print("Previous Thrust: ",self.pre_th)

        # print("self.pre_th : ", self.pre_th)      

        des_th = self.M_bar @ u0 + self.N_cap + self.M_bar @ self.adapt


        #Updating data_out for the bag file
        
        #this line could have issues [check out 187 then]
        self.array2Vector3(curPos, self.data_out.posn)
        self.array2Vector3(u0, self.data_out.u0)
        self.array2Vector3(self.N_cap, self.data_out.N_cap)
        self.array2Vector3(curAccLin, self.data_out.acceleration)
        self.array2Vector3(curAccAng, self.data_out.acceleration)

        self.array2Vector3(errPos, self.data_out.position_error)
        self.array2Vector3(errVel, self.data_out.velocity_error)
        # self.array2Vector3(delTau, self.data_out.delTau)
        

        #Thrust Saturation
        if np.linalg.norm(des_th) > self.max_th:
            des_th = (self.max_th/np.linalg.norm(des_th))*des_th


        #Saving the values of Thrust and posn to be used next cycle
        self.pre_pose = curPos
        self.pre_acc = curAccLin
        self.pre_th = des_th

        return des_th

    def acc2quat(self, des_th, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_th) == 0.0:
            zb_des = np.array([0,0,1])
        else:    
            zb_des = des_th / np.linalg.norm(des_th)
        yb_des = np.cross(zb_des, proj_xb_des) / np.linalg.norm(np.cross(zb_des, proj_xb_des))
        xb_des = np.cross(yb_des, zb_des) / np.linalg.norm(np.cross(yb_des, zb_des))
       
        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def geo_con(self):
        des_th = self.th_des()    
        r_des = self.acc2quat(des_th, 0.0)

        # print("r_des", r_des)
        # print(np.hstack((r_des,np.array([[0,0,0]]).T)))
        
        rot_44 = np.vstack((np.hstack((r_des,np.array([[0,0,0]]).T)), np.array([[0,0,0,1]])))

        quat_des = quaternion_from_matrix(rot_44)
       
        zb = r_des[:,2]
        thrust = self.norm_thrust_const * des_th.dot(zb)
        self.data_out.thrust = thrust
        
        thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        now = rospy.Time.now()
        self.att_cmd.header.stamp = now
        self.thrust_cmd.header.stamp = now
        self.data_out.header.stamp = now
        self.att_cmd.pose.orientation.x = quat_des[0]
        self.att_cmd.pose.orientation.y = quat_des[1]
        self.att_cmd.pose.orientation.z = quat_des[2]
        self.att_cmd.pose.orientation.w = quat_des[3]
        self.thrust_cmd.thrust = thrust
        # print(thrust)
        # print(quat_des)
# 
        # self.data_out.orientation = self.att_cmd.pose.orientation

    def pub_att(self):
        self.geo_con()
        self.thrust_pub.publish(self.thrust_cmd)
        self.att_pub.publish(self.att_cmd)
        self.data_pub.publish(self.data_out)

# Main function
def main(argv):
   
    rospy.init_node('setpoint_node', anonymous=True)
    modes = fcuModes()  #flight modes
    cnt = Controller()  # controller object
    rate = rospy.Rate(30)
    rospy.Subscriber('mavros/state', State, cnt.stateCb)

    rospy.Subscriber('mavros/local_position/odom', Odometry, cnt.odomCb)

    # Subscribe to drone's local position
    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)
    rospy.Subscriber('new_pose', PoseStamped, cnt.newPoseCB)
    rospy.Subscriber('command/trajectory', Mdjt, cnt.multiDoFCb)
    sp_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)

    print("ARMING")
    while not cnt.state.armed:
        modes.setArm()
        cnt.armed = True
        rate.sleep()

    cnt.armed = True
    k=0
    while k<20:
        sp_pub.publish(cnt.sp)
        rate.sleep()
        k = k + 1

    modes.setOffboardMode()
    print("---------")
    print("OFFBOARD")
    print("---------")

    # ROS main loop
    while not rospy.is_shutdown():
        # r_des = quaternion_matrix(des_orientation)[:3,:3]
        # r_cur = quaternion_matrix(cnt.local_quat)[:3,:3]

#--------------------------------------------
        cnt.pub_att()
        rate.sleep()
       

#--------------------------------------------  

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except rospy.ROSInterruptException:
        pass
