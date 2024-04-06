import pickle
import pdb
import numpy as np

# class DefaultDict(dict):
#     def __init__(self, data):
#         super().__init__(data)
#         self.default_value = data.get("default_value", None)

#     def __getitem__(self, key):
#         return self.get(key, self.default_value)

#     def __getattr__(self, key):
#         return self.get(key, self.default_value)  # Retrieve value using get()


# #my_dict_class = DefaultDict({"key1": "value1", "key2": "value2", "default_value": "default_value"})
# #print(my_dict_class["key3"])  # Output: "default_value"


# class DataWrapper:
#     def __init__(self, data):
#         self.__dict__ = data

class PickledDataReader:
    def __init__(self, filename):
        with open(filename, "rb") as f:
            self.data = pickle.load(f)
        self.index = 0  # Initialize index for tracking

    def get_next_data(self):
        if self.index < len(self.data):
            data_item = self.data[self.index]
            self.index += 1
            return data_item
        else:
            return None  # Indicate the end of data

if __name__ == '__main__' :
    filename = "imu_sensor_data.pkl"
    imu_pickled = PickledDataReader(filename)
    #imu_data = DataWrapper(pickle.load(imu_pickled))  # Load and wrap in DataWrapper
    imu_ = imu_pickled.get_next_data().imu
    pdb.set_trace()
    # imu = imu_pickled.get_next_data()
    # print(imu_pickled.imu.accelerometer.x)
    # imu = imu.imu
    # pdb.set_trace()
    # imu_f = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]).reshape(3, 1)
    # imu_w = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]).reshape(3, 1)
    # pdb.set_trace()
    