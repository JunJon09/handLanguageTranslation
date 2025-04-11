import h5py
create_path = "../../hdf5/test_data/003.hdf5"
# sample_path = "../../data/dataset_top10/2044.hdf5"
with h5py.File(create_path, "r") as fread:
    keys = fread.keys()
    print("Groups:", keys)
    print(len(keys))
    count = 0
    for key in keys:
        data = fread[key]
        print("Data in a group:", data.keys())
        feature = data["feature"][:]
        token = data["token"][:]
        print(feature.shape)
        count += len(token)
        print((token), key)
    print(count / len(keys))