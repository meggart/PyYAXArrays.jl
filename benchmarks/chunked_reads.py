import xarray as xr
import zarr
import numpy as np

import pyperf # to run benchmarks
import timeit

data_url = "https://its-live-data.s3-us-west-2.amazonaws.com/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"

dataset = xr.open_zarr(data_url)

v = dataset["v"]

SUITE = {}

SUITE["single chunk read"] = timeit.timeit(
    stmt = "output_storage[:, :, :] = v[0:(v.chunks[0][0]), 0:(v.chunks[1][0]), 0:(v.chunks[2][0])]",
    globals = {"v": v, "output_storage": np.zeros((v.chunks[0][0], v.chunks[1][0], v.chunks[2][0]))},
    number = 50,
) / 50

SUITE["hundred chunk read contiguous"] = timeit.timeit(
    stmt = "output_storage[:, :] = v[:, :, 0]",
    globals = {"v": v, "output_storage": np.zeros((sum(v.chunks[0]), sum(v.chunks[1])))},
    number = 10,
) / 10

SUITE["random access"] = timeit.timeit(
    stmt = "v[x, y, z]",
    globals = {"v": v, "x": np.random.randint(0, sum(v.chunks[0]), 1000), "y": np.random.randint(0, sum(v.chunks[1]), 1000), "z": np.random.randint(0, sum(v.chunks[2]), 1000)},
    number = 10,
) / 10

import json
with open("python.json", "w") as f:
    json.dump(SUITE, f)