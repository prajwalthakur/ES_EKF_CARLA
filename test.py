import pickle

class ImuData:
    def __init__(self, accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z):
        self.imu = {
            "accelerometer": {
                "x": accelerometer_x,
                "y": accelerometer_y,
                "z": accelerometer_z,
            },
            "gyroscope": {
                "x": gyroscope_x,
                "y": gyroscope_y,
                "z": gyroscope_z,
            },
        }

    def __getattr__(self, item):
        return self.imu[item]  # Delegate attribute access to the nested 'imu' dictionary
    def __getstate__(self):
        # Empty method to satisfy pickling (optional)
        return self.__dict__  # Or return any picklable data here

# Create an ImuData object
imu_data = ImuData(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

# Pickle the object
with open("imu_data.pickle", "wb") as f:
    pickle.dump(imu_data, f)

print("IMU data pickled successfully!")


# Load the pickled object
with open("imu_data.pickle", "rb") as f:
    loaded_imu_data = pickle.load(f)

print("\nLoaded IMU data:")

# Access data using dot notation
print(f"Accelerometer x: {loaded_imu_data.imu.accelerometer.x}")
print(f"Gyroscope y: {loaded_imu_data.imu.gyroscope.y}")
