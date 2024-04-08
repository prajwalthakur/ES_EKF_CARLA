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
from pickle_class import imu_pickle,gnss_pickle,gt_pickle
import math
#utility function
def calculate_sides(hypotenuse,angle):
    
    
    angle_radians = math.radians(angle)
    opp_side = hypotenuse*math.sin(angle_radians)
    adj_side = hypotenuse*math.cos(angle_radians)
    return opp_side ,adj_side

def get_spectator_pos(vehicle):
    # subtract the delta x and y to be behind
    meter_distance = 10
    vehicle_transform = vehicle.get_transform()
    y,x = calculate_sides(meter_distance,vehicle_transform.rotation.yaw)
    spectator_pos = carla.Transform(vehicle_transform.location + carla.Location(x=-x,y=-y,z=20)
                                    , carla.Rotation(yaw= vehicle_transform.rotation.yaw,pitch=-25))
    # loc = carla.Location(x=vehicle_transform.location.x,y=vehicle_transform.location.y,z=15)
    # spectator_pos = carla.Transform(loc
    #                                 , carla.Rotation(yaw= 180.0,pitch=-90.0))
    return spectator_pos


def main(args):
    if_save_data = False
    gt_trajc = []
    imu_sensor_data = []
    gnss_sensor_data = []
    try:
        client = carla.Client(args.host, args.port)
        #client = carla.Client('loacalhost',2000)
        client.set_timeout(10.0)
        world = client.get_world()
        # settings = world.get_settings()
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05
        # world.apply_settings(settings)
        spawn_point  = random.choice(world.get_map().get_spawn_points())
        
        #Create a car object
        car = Car(world,client,spawn_point)
        spectator = world.get_spectator()
        #world_snapshot = world.wait_for_tick() 
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
            spectator_pos = get_spectator_pos(car.vehicle)
            spectator.set_transform(spectator_pos)
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
                imu_copy = imu_pickle(imu)
                if if_save_data:
                    gt_trajc.append(gt_loc_copy)
                    imu_sensor_data.append(imu_copy)
            
            
            # ES_EKF correction with the gnss/gps reading                   
            # Get gps reading @ 5Hz
            if sensors['gnss'] is not None and sensors['imu'] is not None: 
                gnss = sensors['gnss']
                es_ekf.state_correction_with_gnss(gnss)
                timestamp  = sensors['imu'].timestamp
                gnss_copy = gnss_pickle(gnss,timestamp)
                if if_save_data:
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
        if if_save_data:
            pickle.dump(imu_sensor_data, open('pkl_data/imu_sensor_data.pkl','wb'))
            pickle.dump(gt_trajc, open('pkl_data/gt_trajectory.pkl','wb'))
            pickle.dump(gnss_sensor_data, open('pkl_data/gnss_sensor_data.pkl','wb'))
        print('Exiting Visualizer')
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
            
            
            
        
        