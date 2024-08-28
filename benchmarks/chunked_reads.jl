# WARNING: Python must always be loaded before Julia, because it breaks if Julia's OpenSSL
# is loaded first.  There is not a good way to fix this, so we just have to live with it.
using PythonCall # for Python testing
using PyYAXArrays # Xarrays wrapped in Julia 🏴‍☠️
using FSSpec # fsspec wrapped in Julia 🤯
const xr = PyYAXArrays.xr[] # this is a reference to the xarray module in Python

using BenchmarkTools, Chairmarks, CairoMakie, SwarmMakie, Statistics, DiskArrays # benchmarking and plotting
#using ProfileView#, Cthulhu # profile and figure out where type instability happens

using YAXArrays, Zarr # the Julia end of the spectrum

const SUITE = BenchmarkGroup() # instantiate a toplevel benchmark group which all benchmarks will be stored in

# Set up the data

# data_url = "https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1" # not working with Python?!
data_url = "https://its-live-data.s3-us-west-2.amazonaws.com/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"
# data_url = "s3://its-live-data/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"

yax_array = YAXArrays.open_dataset(Zarr.zopen(data_url, consolidated = true))

py_array = YAXArrays.open_dataset(xr.open_zarr(data_url, decode_times=false))

fs_array = YAXArrays.open_dataset(Zarr.zopen(FSSpec.FSStore("s3://its-live-data/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"), consolidated = true))

# Load a chunk
# Load 100 contiguous chunks
# Load partial chunks
# Totally random access (this is also partial chunks)

# This function has to do the benchmarking internally, because chunk structures can change across different arrays and chunking structures.
function benchmark_single_chunk_read(A)
    chunk_idxs = first(DiskArrays.eachchunk(A))
    aout = zeros(Union{Missing, Float64}, chunk_idxs...)
    @benchmark DiskArrays.readblock!(parent($A #=A must be a YAXArray for this to work=#), $aout, $(chunk_idxs)...)
end

SUITE["single chunk read"]["YAXArrays"] = benchmark_single_chunk_read(yax_array["v"])
SUITE["single chunk read"]["PyYAXArrays"] = benchmark_single_chunk_read(py_array["v"])
SUITE["single chunk read"]["FSSpec"] = benchmark_single_chunk_read(fs_array["v"])

# Load 100 contiguous chunks.  For the MUR SST data, we'll do this by loading 

function load_spatial_contiguous_chunks!(array, out)
    out .= array[:, :, 1]
end

for (lang, array) in [("YAXArrays", yax_array["v"]), ("PyYAXArrays", py_array["v"]), ("FSSpec", fs_array["v"])]
    out = zeros(Union{Missing, Float64}, size(array, 1), size(array, 2))
    SUITE["hundred chunk read contiguous"]["$lang"] = @benchmark load_spatial_contiguous_chunks!($(array), $(out)) seconds=10
end

#= performance profiling
const ___array = py_array["v"]
const ___out = zeros(Union{Missing, Float64}, size(___array, 1), size(___array, 2))

ProfileView.@profview load_spatial_contiguous_chunks!(___array, ___out)

=#

function random_access(array, inds)
    array[inds]
end

for (lang, array) in [("YAXArrays", yax_array["v"]), ("PyYAXArrays", py_array["v"]), ("FSSpec", fs_array["v"])]
    out = zeros(Union{Missing, Float64}, 1000)
    indices = CartesianIndex.(rand(1:size(array, 1), 1000), rand(1:size(array, 2), 1000), rand(1:size(array, 3), 1000))
    SUITE["random access"]["$lang"] = @benchmark $(out) .= $(array)[$(indices)] seconds=10
end

function medium_read(array)
    collect(array[:, :, 1:15]) # TODO: does this count as chunking?
end

SUITE["medium read"]["YAXArrays"] = @benchmark medium_read($(yax_array["v"])) seconds=10
SUITE["medium read"]["PyYAXArrays"] = @benchmark medium_read($(py_array["v"])) seconds=10
SUITE["medium read"]["FSSpec"] = @benchmark medium_read($(fs_array["v"])) seconds=10

function some_friendly_computation(array)
    mapslices(mean, array, dims="Ti")
end

SUITE["medium computation"]["YAXArrays"] = @benchmark some_friendly_computation($(yax_array["v"][1:50, 1:50, :])) seconds=10
SUITE["medium computation"]["PyYAXArrays"] = @benchmark some_friendly_computation($(py_array["v"][1:50, 1:50, :])) seconds=10
SUITE["medium computation"]["FSSpec"] = @benchmark some_friendly_computation($(fs_array["v"][1:50, 1:50, :])) seconds=10

function timeseries_access(array)
    mapslices(mean, array, dims="Ti")
end

SUITE["timeseries access"]["PyYAXArrays"] = @benchmark timeseries_access($(yax_array["v"])) seconds=20
SUITE["timeseries access"]["YAXArrays"] = @benchmark timeseries_access($(py_array["v"])) seconds=20
SUITE["timeseries access"]["FSSpec"] = @benchmark timeseries_access($(fs_array["v"])) seconds=20



function fake_trial(time::Real)
    return BenchmarkTools.Trial(
        BenchmarkTools.Parameters(),
        Float64[time],
        Float64[0],
        0,
        0
    )
end


function load_python_benchmark_suite!(file::String)
    global SUITE

    dict = Dict{Symbol, Float64}(JSON3.read(read(file, String)))

    for (name, time) in pairs(dict)
        SUITE[string(name)]["Python"] = fake_trial(time #= in seconds =# * 1e9 #= to nanoseconds =#)
    end
end

load_python_benchmark_suite!("python.json")

