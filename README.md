# Error state Extended Kalman Filter for Sensor Fusion

##Three ways to run the project

## Steps to run the project with CARLA
##### Requirements
- CARLA (follow the step mentioned here : [install carla](https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation:~:text=B.-,Package%20installation,-CARLA%20repository))
- python3
- pip3 install -r requirements.txt
##### Steps to run
- run carla : ./CarlaUE4.sh -prefernvidia -quality-level=Low
- python3 main_sim.py

## Steps to run the project without CARLA and with recorded sensor data and with animation plot
In this case sensor-data has been pre-recorded  and can now be traversed iteratively . 
### requirements
- python3 
- pip3 install -r requirements.txt
### steps to rrun
- python3 main_recorded.py

