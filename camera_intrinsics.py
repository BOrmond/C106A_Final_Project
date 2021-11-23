import h5py
filename = "camera_intrinsics.h5"

with h5py.File(filename, "r") as data:

    #fisheye intrinsics
    fish_K = data["fisheye"]["K"]
    fish_distortion = data["fishey"]["distortion"]

    #leftpillar intrinsics
    lp_K = data["leftpillar"]["K"]
    lp_distortion = data["leftpillar"]["distortion"]

    #leftrepeater intrinsics
    lr_K = data["leftrepeater"]["K"]
    lr_distortion = data["leftrepeater"]["distortion"]

    #main intrinsics
    main_K = data["main"]["K"]
    main_distortion = data["main"]["distortion"]

    #rightpillar intrinsics
    rp_K = data["rightpillar"]["K"]
    rp_distortion = data["rightpillar"]["distortion"]

    #leftrepeater intrinsics
    rr_K = data["rightrepeater"]["K"]
    rr_distortion = data["rightrepeater"]["distortion"]
