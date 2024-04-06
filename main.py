import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import cv2
import time

from multiprocessing import Queue, Value, Process
from ctypes import c_bool

from car import Car
from visualizer import visualizer

from util import destroy_queue
from errorState_KF import es_ekf

def main():
    try:
        client = carla.Client('loacalhost',2000)
        client.set_timeout(10.0)
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        settings.fixed_delta_seconds = None # Set a variable time-step
        world.apply_settings(settings)
        spawn_point  = random.choice(world.get_map().get_spawn_points())
        
        #Create a car object
        car = Car(world,client,spawn_point)
        print(" created a car object ")
        world_msg_queue = Queue()
        quit  = Value(c_bool,False)
        proc = Process(target = visualizer , args = (world_msg_queue,quit))
        proc.daemon = True
        proc.start()
        
        visual_fps = 3
        last_ts = time.Time()
        # visual message
        world_msg = dict()        
        # Drive the car around and get sensor readings
        while True:
            world.tick()
            frame = world.get_snapshot().frame
            sensors = car.get_sensor_readings(frame)
            #get ground truth value of vehicle to compare ES-EKF estimate and Ground truth Estimate
            gt_location = car.get_location()
            world_msg['gt_traj'] =  [gt_location.x,gt_location.y,gt_location.z]
            
            if not es_ekf.is_initialized():
                # to initialize the the KF with current true state (only for first time)
                es_ekf.initialize_with_true_state(gt_location)
                
                # to initialize the KF with the GNSS data 
                
                # if sensors['gnss'] is not None :
                #     gnss = sensors['gnss']
                #     es_ekf.initialize_with_gnss(gnss)
            
            # EKF Predicition -- imu is running at 200Hz much faster than GNSS update 
            #get imu reading
            if sensors['imu'] is not None:
                imu = sensors['imu']
                accel = imu.accelerometer
                gyro = imu.gyroscope
                world_msg['imu'] = [accel.x,accel.y,accel.z,
                                     gyro.x, gyro.y, gyro.z]
                
                es_ekf.predict_state_with_imu(imu)
            
            
            # ES_EKF correction with the gnss/gps reading                   
            # Get gps reading
            if sensors['gnss'] is not None :
                gnss = sensors['gnss']
                world_msg['gnss'] = [gnss.x,gnss.y,gnss.z]
                es_ekf.correct_state_with_gnss(gnss)
            
            # Get estimated location
            world_msg['est_traj'] = es_ekf.get_location()
            world_msg_queue.put(world_msg) # enqueue the current msg 
    finally :
        print('Existing Visualizer')
        quit.value = True
        destroy_queue(world_msg)
        print('destroying the car object')
        car.destroy()
        print('done')
        
    
if __name__ == '__main__' :
    main()
            
            
            
        
        