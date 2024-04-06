# Author: Shing-Yan Loo (yan99033@gmail.com)
# This script spawns a car at a random spawn point in the map, runs
# Kalman filter, and at the same time visualizes the sensor signals 
# The list of sensors used are as follows:
# - camera (only for visualization, not being used for localization)
# - IMU
# - GNSS

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
# import cv2
import time

from multiprocessing import Queue, Value, Process
from ctypes import c_bool

from car import Car
from errorState_KF  import error_state_ekf
from visualizer import visualizer

from util import destroy_queue

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        spawn_point = random.choice(world.get_map().get_spawn_points())

        # EKF
        ekf = error_state_ekf()

        # Create a car object
        car = Car(world, client, spawn_point)
        print('created a car object')

        # Visualizer
        visual_msg_queue = Queue()
        quit = Value(c_bool, False)
        proc = Process(target =visualizer, args=(visual_msg_queue, quit))
        proc.daemon = True
        proc.start()

        # In case Matplotlib is not able to keep up the pace of the growing queue,
        # we have to limit the rate of the items being pushed into the queue
        visual_fps = 3
        last_ts = time.time()

        # Drive the car around and get sensor readings
        while True:
            world.tick()
            frame = world.get_snapshot().frame

            # Get sensor readings
            sensors = car.get_sensor_readings(frame)

            gt_location = car.get_location()
            visual_msg = dict()
            visual_msg['gt_traj'] =  [gt_location.x,gt_location.y,gt_location.z]
            
            if not ekf.is_initialized():
                # to initialize the the KF with current true state (only for first time)
                ekf.initialize_with_true_state(gt_location)
            # EKF initialization
            # Don't run anything else before EKF is initialized
            # if not ekf.is_initialized():
            #     if sensors['gnss'] is not None:
            #         ekf.initialize_with_gnss(sensors['gnss'])
                    
            #     continue

            # EKF prediction
            if sensors['imu'] is not None:
                ekf.predict_state_with_imu(sensors['imu'])

            # EKF correction
            if sensors['gnss'] is not None:
                ekf.state_correction_with_gnss(sensors['gnss'])
            
            # Limit the visualization frame-rate
            if time.time() - last_ts < 1. / visual_fps:
                continue
            
            # timestamp for inserting a new item into the queue
            last_ts = time.time()

            # visual message
            

            # Get ground truth vehicle location
            gt_location = car.get_location()
            visual_msg['gt_traj'] = [gt_location.x, gt_location.y, gt_location.z] # round(x, 1)

            # Get estimated location
            visual_msg['est_traj'] = ekf.get_location()

            # Get imu reading
            if sensors['imu'] is not None:
                imu = sensors['imu']
                accelero = imu.accelerometer
                gyroscop = imu.gyroscope
                visual_msg['imu'] = [accelero.x, accelero.y, accelero.z, 
                                     gyroscop.x, gyroscop.y, gyroscop.z]

            # Get gps reading
            if sensors['gnss'] is not None:
                gnss = sensors['gnss']
                visual_msg['gnss'] = [gnss.x, gnss.y, gnss.z]            

            visual_msg_queue.put(visual_msg)
            print("gt-trajectory=", visual_msg['gt_traj'],"----", "est-trajectory", visual_msg['est_traj'])

    finally:
        print('Exiting visualizer')
        quit.value = True
        destroy_queue(visual_msg_queue)

        print('destroying the car object')
        car.destroy()

        print('done')



if __name__ == '__main__':
    main()