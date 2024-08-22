using BenchmarkTools, Chairmarks, CairoMakie, SwarmMakie, Statistics, DiskArrays # benchmarking and plotting

using YAXArrays, Zarr # the Julia end of the spectrum

using PythonCall # for Python testing

using PyYAXArrays # Xarrays wrapped in Julia 🏴‍☠️
const xr = PyYAXArrays.xr[]

const SUITE = BenchmarkGroup()

# Set up the data

# data_url = "https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1" # not working with Python?!
data_url = "https://its-live-data.s3-us-west-2.amazonaws.com/datacubes/v2/N00E020/ITS_LIVE_vel_EPSG32735_G0120_X750000_Y10050000.zarr"

yax_array = YAXArrays.open_dataset(Zarr.zopen(data_url, consolidated = true))

py_array = YAXArrays.open_dataset(xr.open_zarr(data_url, decode_times=false))

# Load a chunk
# Load 100 contiguous chunks
# Load partial chunks
# Totally random access (this is also partial chunks)

# This function has to do the benchmarking internally, because chunk structures can change across different arrays and chunking structures.
function benchmark_single_chunk_read(A)
    chunk_idxs = first(DiskArrays.eachchunk(A))
    aout = zeros(chunk_idxs...)
    @benchmark DiskArrays.readblock!(parent($A #=A must be a YAXArray for this to work=#), $aout, $(chunk_idxs)...)
end

SUITE["single chunk read"]["Julia"] = benchmark_single_chunk_read(yax_array["analysed_sst"])
SUITE["single chunk read"]["Python"] = benchmark_single_chunk_read(py_array["analysed_sst"])

# Load 100 contiguous chunks.  For the MUR SST data, we'll do this by loading 

function load_spatial_contiguous_chunks!(array, out)
    out .= array[:, :, 1]
end

for (lang, array) in [("Julia", yax_array["analysed_sst"]), ("Python", py_array["analysed_sst"])]
    out = zeros(size(array)[1:2]...)
    SUITE["hundred chunk read contiguous"]["$lang"] = @benchmark load_spatial_contiguous_chunks!($(array), $(out)) seconds=10
end

function medium_read(array)
    collect(array[1:1000, 1:1000, 1:15]) # TODO: does this count as chunking?
end


SUITE["medium read"]["Julia"] = @benchmark medium_read($(yax_array["analysed_sst"])) seconds=10
SUITE["medium read"]["Python"] = @benchmark medium_read($(py_array["analysed_sst"])) seconds=10

function some_friendly_computation(array)
    mapslices(mean, array, dims="Ti")
end

SUITE["medium computation"]["Julia"] = @benchmark some_friendly_computation($(yax_array["analysed_sst"][1:1000, 1:1000, 1:15])) seconds=10
SUITE["medium computation"]["Python"] = @benchmark some_friendly_computation($(py_array["analysed_sst"][1:1000, 1:1000, 1:15])) seconds=10

function timeseries_access(array)
    mapslices(mean, array, dims="Ti")
end

SUITE["timeseries access"]["Python"] = @benchmark timeseries_access($(yax_array["analysed_sst"])) seconds=20
SUITE["timeseries access"]["Julia"] = @benchmark timeseries_access($(py_array["analysed_sst"])) seconds=20



SUITE["random spatial access"]["Python"] = @benchmark benchmark_random_spatial_access($(yax_array["analysed_sst"])) seconds=20
SUITE["random spatial access"]["Julia"] = @benchmark benchmark_random_spatial_access($(py_array["analysed_sst"])) seconds=20