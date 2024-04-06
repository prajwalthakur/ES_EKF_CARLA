import numpy as np
import pdb
import copy
class imu_pickle:
    def __init__(self, imu):
        self.imu = {}
        self.imu = {
            "accelerometer": {
                "x": copy.deepcopy(imu.accelerometer.x),
                "y": copy.deepcopy(imu.accelerometer.y),
                "z": copy.deepcopy(imu.accelerometer.z),
            },
            "gyroscope": {
                "x": copy.deepcopy(imu.gyroscope.x),
                "y": copy.deepcopy(imu.gyroscope.y),
                "z": copy.deepcopy(imu.gyroscope.z),
            },
            "timestamp":copy.deepcopy(imu.timestamp)
        }
        
        
class gnss_pickle:
    def __init__(self, gnss,timestamp):
        self.gnss = {}
        self.gnss = {
            "gnss": {
                "x": copy.deepcopy(gnss.x),
                "y": copy.deepcopy(gnss.y),
                "z": copy.deepcopy(gnss.z),
            },
            "timestamp":timestamp #copy.deepcopy(gnss.timestamp)
        }
        
class gt_pickle:
    def __init__(self, gt):
        self.gt_location = {}
        self.gt_location = {
            "gt_location": {
                "x": copy.deepcopy(gt.x),
                "y": copy.deepcopy(gt.y),
                "z": copy.deepcopy(gt.z),
            }
        }