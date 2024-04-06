import glob
import os
import sys
import pdb

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time

from multiprocessing import Queue, Value, Process
from ctypes import c_bool

from car import Car
from visualizer import visualizer

from util import destroy_queue
from errorState_KF import error_state_ekf
import logging
import argparse
import pickle
import copy
from pickle_class import imu_pickle,gnss_pickle,gt_pickle
def main(args):
    gt_trajc = []
    imu_sensor_data = []
    gnss_sensor_data = []
    try:
        client = carla.Client(args.host, args.port)
        #client = carla.Client('loacalhost',2000)
        client.set_timeout(10.0)
        world = client.get_world()
        #settings = world.get_settings()
        #settings.synchronous_mode = True
        #settings.fixed_delta_seconds = 0.05
        #world.apply_settings(settings)
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
        last_ts = time.time()
        # visual message
        world_msg = dict()        
        # Drive the car around and get sensor readings
        es_ekf = error_state_ekf()
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
                es_ekf.predict_state_with_imu(imu)
                gt_loc_copy = gt_pickle(gt_location)
                gt_trajc.append(gt_loc_copy)
                imu_copy = imu_pickle(imu)
                imu_sensor_data.append(imu_copy)
            
            
            # ES_EKF correction with the gnss/gps reading                   
            # Get gps reading @ 5Hz
            if sensors['gnss'] is not None and sensors['imu'] is not None: 
                gnss = sensors['gnss']
                es_ekf.state_correction_with_gnss(gnss)
                timestamp  = sensors['imu'].timestamp
                gnss_copy = gnss_pickle(gnss,timestamp)
                gnss_sensor_data.append(gnss_copy)

            # Limit the visualization frame-rate
            if time.time() - last_ts < 1. / visual_fps:
                continue
            
            # timestamp for inserting a new item into the queue
            last_ts = time.time()




            if sensors['imu'] is not None:
                imu = sensors['imu']
                accel = imu.accelerometer
                gyro = imu.gyroscope
                world_msg['imu'] = [accel.x,accel.y,accel.z,
                                     gyro.x, gyro.y, gyro.z]

            
            
            # ES_EKF correction with the gnss/gps reading                   
            # Get gps reading
            if sensors['gnss'] is not None :
                gnss = sensors['gnss']
                world_msg['gnss'] = [gnss.x,gnss.y,gnss.z]
            
            # Get estimated location
            world_msg['est_traj'] = es_ekf.get_location()
            world_msg_queue.put(world_msg) # enqueue the current msg
            print("gt_location=",gt_location,"---","est_location=",world_msg['est_traj'])
    except KeyboardInterrupt :
        pickle.dump(imu_sensor_data, open('imu_sensor_data.pkl','wb'))
        pickle.dump(gt_trajc, open('gt_trajectory.pkl','wb'))
        pickle.dump(gnss_sensor_data, open('gnss_sensor_data.pkl','wb'))
        print('Existing Visualizer')
        quit.value = True
        destroy_queue(world_msg_queue)
        print('destroying the car object')
        car.destroy()
        print('done')
        
    
if __name__ == '__main__' :
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    # Parse arguments
    args = argparser.parse_args()
    args.description = argparser.description
    # Print server information
    #log_level = logging.DEBUG if args.debug else logging.INFO
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    main(args)
            
            
            
        
        