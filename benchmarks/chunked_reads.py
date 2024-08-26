import xarray as xr
import zarr
import numpy as np

import timeit # to run benchmarks

data_url = "https://its-live-data.s3-us-west-2.amazonaws.com/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"

dataset = xr.open_zarr(data_url)

v = dataset["v"]

single_chunk_read_time = timeit.timeit(
    stmt = "output_storage[:, :, :] = v[0:(v.chunks[0][0]), 0:(v.chunks[1][0]), 0:(v.chunks[2][0])]",
    globals = {"v": v, "output_storage": np.zeros((v.chunks[0][0], v.chunks[1][0], v.chunks[2][0]))},
    number = 50,
) / 50

hundred_chunk_contiguous_read_time = timeit.timeit(
    stmt = "output_storage[:, :] = v[:, :, 0]",
    globals = {"v": v, "output_storage": np.zeros((sum(v.chunks[0]), sum(v.chunks[1])))},
    number = 10,
) / 10

random_access_time = timeit.timeit(
    stmt = "v[x, y, z]",
    globals = {"v": v, "x": np.random.randint(0, sum(v.chunks[0]), 1000), "y": np.random.randint(0, sum(v.chunks[1]), 1000), "z": np.random.randint(0, sum(v.chunks[2]), 1000)},
    number = 10,
) / 10
