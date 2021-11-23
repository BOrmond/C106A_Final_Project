import h5py
filename = "extrinsics.h5"

with h5py.File(filename, "r") as data:

    #Extrinsic Matrices
    fisheye = data["fisheye"][:]
    leftpillar = data["leftpillar"][:]
    leftrepeater = data["leftrepeater"][:]
    lidar0 = data["lidar0"][:]
    lidar1 = data["lidar1"][:]
    lidar2 = data["lidar2"][:]
    lidar3 = data["lidar3"][:]
    lidar4 = data["lidar4"][:]
    lidar5 = data["lidar5"][:]
    #lidar6 missing for some reason?
    lidar7 = data["lidar7"][:]
    lidar8 = data["lidar8"][:]
    main = data["main"][:]
    rightpillar = data["rightpillar"][:]
    rightrepeater = data["rightrepeater"][:]
    