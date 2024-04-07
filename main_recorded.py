
import time
from multiprocessing import Queue, Value, Process
from ctypes import c_bool
from visualizer import visualizer
from util import destroy_queue
from errorState_KF import error_state_ekf
from load_pickle import PickledDataReader
def main():
    # loading pkl data collected by running main_sim.py 
    gt_trajc = PickledDataReader("pkl_data/gt_trajectory.pkl")
    imu_sensor_data =PickledDataReader("pkl_data/imu_sensor_data.pkl")
    gnss_sensor_data = PickledDataReader("pkl_data/gnss_sensor_data.pkl")
    try:
        world_msg_queue = Queue()
        quit  = Value(c_bool,False)
        proc = Process(target = visualizer , args = (world_msg_queue,quit))
        proc.daemon = True
        proc.start()
        world_msg = dict()        
        #ES-EKF initialization
        es_ekf = error_state_ekf()
        gnss_i = 0
        for index,data in enumerate(imu_sensor_data.data):           
            #get ground truth value of vehicle to compare ES-EKF estimate and Ground truth Estimate
            gt_location = gt_trajc.get_next_data().gt_location
            world_msg['gt_traj'] =  [gt_location['gt_location']['x'],gt_location['gt_location']['y'],gt_location['gt_location']['z']]
            if not es_ekf.is_initialized():
                # to initialize the the KF with current true state (only for first time)
                es_ekf.initialize_with_true_state(gt_location,recorded=True)
            
            # get imu data  : IMU is running at 200Hz
            imu = imu_sensor_data.get_next_data().imu
            accel = imu['accelerometer']
            gyro = imu['gyroscope']
            world_msg['imu'] = [ accel['x'],accel['y'],accel['z'],
                                    gyro['x'], gyro['y'], gyro['z'] ]           
            # model  Predicition update
            es_ekf.predict_state_with_imu(imu,recorded=True)
            
            # state correction with GNSS data 
            gnss_at = gnss_sensor_data.data_at(gnss_i).gnss
            if gnss_at['timestamp'] == imu['timestamp'] :
                gnss = gnss_sensor_data.get_data_at(gnss_i).gnss
                gnss_i+=1
                es_ekf.state_correction_with_gnss(gnss,recorded=True)
                world_msg['gnss'] = [gnss['gnss']['x'],gnss['gnss']['y'],gnss['gnss']['z']]
                        
            # Get estimated location
            world_msg['est_traj'] = es_ekf.get_location()
            world_msg_queue.put(world_msg) # enqueue the current msg
            print("gt_location=",gt_location,"---","est_location=",world_msg['est_traj'])
            time.sleep(0.12)
            
    except KeyboardInterrupt :
        print('Existing Visualizer')
        quit.value = True
        destroy_queue(world_msg_queue)
        print('destroying the car object')
        print('done')
        
    
if __name__ == '__main__' :
    main()
            
            
            
        
        