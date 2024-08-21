using BenchmarkTools, Chairmarks, CairoMakie, SwarmMakie, Statistics # benchmarking and plotting

using YAXArrays, Zarr # the Julia end of the spectrum

using PythonCall # for Python testing

using PyYAXArrays # Xarrays wrapped in Julia 🏴‍☠️
const xr = PyYAXArrays.xr[]

const SUITE = BenchmarkGroup()

# Set up the data

data_url = "https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1" # not working with Python?!
# data_url = "https://its-live-data.s3.amazonaws.com/test_datacubes/v02_latest/ALA/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr"

yax_array = YAXArrays.open_dataset(Zarr.zopen(data_url, consolidated = true))

py_array = YAXArrays.open_dataset(xr.open_zarr(data_url, decode_times=false))

function benchmark_medium_read(array)
    collect(array[1:360, 1:170, 1:50]) # TODO: does this count as chunking?
end

function benchmark_computation(array)
    mapslices(mean, array, dims="Ti")
end

@benchmark benchmark_medium_read($(yax_array["analysed_sst"])) seconds=20
@benchmark benchmark_medium_read($(py_array["analysed_sst"])) seconds=20

@benchmark benchmark_computation($(yax_array["analysed_sst"])) seconds=20
@benchmark benchmark_computation($(py_array["analysed_sst"])) seconds=20
