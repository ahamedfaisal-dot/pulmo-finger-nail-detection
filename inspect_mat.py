import h5py
import numpy as np

print("Trying to read TM.mat as HDF5 (MATLAB v7.3 format)...")
try:
    with h5py.File('TM.mat', 'r') as f:
        def print_hdf5(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} | shape={obj.shape} dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group:   {name}")
        
        print("Top-level keys:", list(f.keys()))
        f.visititems(print_hdf5)
except Exception as e:
    print(f"HDF5 read failed: {e}")
    print()
    print("Trying scipy mat73...")
    try:
        import mat73
        data = mat73.loadmat('TM.mat')
        print("Keys:", list(data.keys()))
    except ImportError:
        print("mat73 not installed")
    except Exception as e2:
        print(f"mat73 failed: {e2}")
