import h5py


filename = './ROIs/CSI1/h5/CSI1_ROIs_TR1.h5'

data={}

with h5py.File(filename, 'r') as f:
    keys = list(f.keys())
    for i in keys:
        data[i] = list(f[i])