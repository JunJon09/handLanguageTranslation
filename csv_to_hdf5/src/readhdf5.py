import h5py
create_path = "../../../hdf5/gafs_dataset_very_small/0.hdf5"
sample_path = "../../data/dataset_top10/2044.hdf5"
with h5py.File(create_path, "r") as fread:
    keys = fread.keys()
    print("Groups:", keys)
    for key in keys:
        data = fread[key]
        print("Data in a group:", data.keys())
        feature = data["feature"][:]
        token = data["token"][:]
        print(feature.shape)
        print(token)