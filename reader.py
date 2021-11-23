import h5py
filename = "odometry.h5"

with h5py.File(filename, "r") as data:
    
    for group in data.keys() :
        print (group)
        print(data[group])
        for dset in (data[group]).keys() :
            print (dset)
            ds_data = data[group][dset] # returns HDF5 dataset object
            print (ds_data)
            print (ds_data.shape, ds_data.dtype)
            arr = data[group][dset][:] # adding [:] returns a numpy array
            print (arr.shape, arr.dtype)
            print (arr)