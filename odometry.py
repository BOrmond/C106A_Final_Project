import h5py
filename = "odometry.h5"

with h5py.File(filename, "r") as data:
    
    #Linear acceleration (m/s^2) and timestamps
    acc_timestamps = data["acc"]["timestamps"][:]
    lin_acc = data["acc"]["linear_acceleration_mps2"][:]

    #GPS position and timestamps
    gps_timestamps = data["gps"]["timestamps"][:]
    gps_pos = data["gps"]["ecef_position"][:]

    #Angular Velocity (rad/s) and timestamps
    angvel_timestamps = data["gyr"]["timestamps"][:]
    angvel = data["gyr"]["angular_velocity_radps"][:]

    #Speed (m/s) and timestamps
    speed_timestamps = data["speed"]["timestamps"][:]
    speed = data["speed"]["scalar_speed_mps"][:]
