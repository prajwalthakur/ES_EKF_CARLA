
import numpy as np
from matplotlib.pyplot import axis
from rotations import Quaternion,omega,skew_symmetric,angle_normalize

class es_ekf:
    def __init__(self):
        # state of the system (delta-xk = delta-position,delta-velocity and delta-orientation) (3+3+4) #4 quaternion
        # delta-xk = F_{k-1}delta-x_{k-1} + L_{k-1}n_{k-1}
        self.p = np.zeros((3,1))
        self.v = np.zeros((3,1))
        self.q = np.zeros((4,1)) #quaternion (w,x,y,z)
        
        # state covariance-estimate
        self.p_cov = np.zeros((9,9)) # 1 less because no covariance against w=1
        
        #last updated timestep  (timestep when imu get updated)
        self.last_imu_Ts = 0
        
        #parameters
        self.g = np.array([0.0,0.0,-9.81]).reshape((3,1))
        
        #Sensor noise variances
        self.var_imu_acc = 0.01
        self.var_imu_gyro = 0.01
        
        #gnss noise variance
        self.var_gnss = 0.01
        
        # Motion model noise Jacobian
        #imu measure 3 linear-acceleration and 3 angular acceleration
        #L_{K} jacobain ( state wrt to imu-measurement ) = 9x6
        self.L_jac = np.zeros([9, 6])
        
        self.L_jac[3:, :] = np.eye(6)  # motion model noise jacobian this is due to the nois in imu

        # Measurement model Jacobian GNSS
        # H_{K} jacobian (state wrt to gnss-measurement)
        # y_k =  [1 0 0]'xk + vk
        self.H_jac = np.zeros([3, 9])
        self.H_jac[:, :3] = np.eye(3)

        # Initialized
        self.n_gnss_taken = 0
        self.gnss_init_xyz = None
        self.initialized = False

    def is_initialized(self):
        return self.initialized
    
    def initialize_with_true_state(self,gt_location):
        self.p[:,0] = np.array([gt_location.x,gt_location.y,gt_location.z]).reshape((3,1))
        self.q[:, 0] = Quaternion().to_numpy()       # w=1., x=0., y=0., z=0. 
        
        # estimate on p-covariance :
        #low uncertainty in position estimation and high in orientation and velocity
        pos_var = 0.01
        orien_var = 10
        vel_var = 10
        self.p_cov[:3, :3] = np.eye(3) * pos_var
        self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
        self.p_cov[6:, 6:] = np.eye(3) * orien_var
        
    def get_location(self):
        """Return the estimated vehicle location

        :return: x, y, z position
        :rtype: list
        """
        return self.p.reshape(-1).tolist()
    
    def predict_state_with_imu(self,imu):
        # imu output is a (noisy-input) to the error-state-space model with process noise :-|--|>
        
        # IMU acceleration and velocity
        imu_f = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]).reshape(3, 1)
        imu_w = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]).reshape(3, 1)
        
        # IMU sampling time
        delta_t = imu.timestamp - self.last_imu_Ts
        self.last_imu_Ts  = imu.timestamp
        
        
        R = Quaternion(*self.q).to_mat()
        # euler_angle =  Quaternion(*self.q).to_euler()
        # roll,pitch,yaw = euler_angle[0],euler_angle[1],euler_angle[2]
        #d_next = p_current + u*delta_t + 0.5*a*delta_t*delta_t
        self.p = self.p + self.v*delta_t + 0.5*delta_t*delta_t*(R@imu_f + self.g) #eqn-1
        #v_next  = v_current + a*delta_t
        self.v = self.v + delta_t*( R@imu_f + self.g ) #eqn-2
        #complicated used off-the shelf implementation
        self.q = omega(imu_w, delta_t) @ self.q  #eqn-3
        
        
        # Update Covariance
        F = self._calulcate_motion_model_jacobian(R,imu_f,delta_t) #Jacobian of the motion (state-space model) model
        Q = self._imu_noise_var(delta_t)
        R = self._gnss_noise_var()
        self.p_cov = 0.0 #TODO:
        
        
    def _calulcate_motion_model_jacobian(self,R,imu_f,delta_t):
        """Calculate the derivative of motion model function with respect to the 
        corrected predictid previous state , previous conrol , 0 process-noise
        states : x,y,z,w,x,y,z,vx,vy,vz"""
        F = np.eye(9)
        
        F[:3,:3] = np.eye(3)                           #jacobain of the x,y,z wrt to the x,y,z (grad of eqn-1 wrt to x)
        F[:3,3:6] = np.eye(3)*delta_t                  #jacobian of the x,y,z wrt to vx,vy,vz (grad of eqn-1 wrt to v)
        F[:3,6:9] = np.zeros(3)                       #jacobian of the x,y,z wrt to qx,qy,qz (grad of eqn-1 wrt to R@imu_f) (grad of (x,y,z) wrt to orientation)
        
        
        #grad of speed (vx,vy,vz) i.e ( grad of equaton-2 )  wrt ...
        F[3:6,:3]  =  np.zeros(3)
        F[3:6,3:6] =  np.eye(3)  #wrt to (vx,vy,z=vz)
        F[3:6,6:9] =  -skew_symmetric(R@imu_f)*delta_t 


        # grad of ( wx,wy,wz ) i.e grad of eqn-3 wrt ...
        F[6:9,:3]  = np.zeros(3)  #wrt to (x,y,z)
        F[6:9,3:6] = np.zeros(3)  #wrt to (vx,vy,vz)
        F[6:9,6:9] =  np.ones(3)  #wrt to (qx,qy,qz)
    
    
    def _imu_noise_var(self,delta_t):
        #IMU measurement model : IMU sensor output  w(t) = 
        """Calculate the IMU noise according to the pre-defined sensor profile"""
        
        Q = delta_t*delta_t*np.diag(np.hstack((np.ones(3,)*self.var_imu_acc,np.ones(3,)*self.var_imu_gyro)))
        
        return Q
    
    
    def _gnss_noise_var(self,delta_t):
        #GNSS measurement model : IMU sensor output  w(t) = 
        """Calculate the GNSS noise according to the pre-defined sensor profile"""
        
        R = np.diag(np.ones(3,)*self.var_gnss)
        
        return R
    
    def state_correction_with_gnss(self,gnss,R):
        #y_k = h(x_k) + vk
        # d_{yk} = H_kd_{xk} + M_kv_k = [1 0 0]d_{xk} + vk
        
        x = gnss.x
        y = gnss.y
        z = gnss.z
        
        #kalman gain
        K = self.p_cov@self.H_jac.T@( np.linalg.inv(self.H_jac@self.p_cov@self.H_jac.T + R) )
        
        # compute the error state (ES-EKF)
        delta_x = K@(np.array([x, y, z])[:, None] - self.p)