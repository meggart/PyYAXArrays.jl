# PyYAXArrays

[![Build Status](https://github.com/meggart/PyYAXArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/meggart/PyYAXArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)


### Example

````julia
using PyYAXArrays, YAXArrays

xr = PyYAXArrays.xr[]
python_dataset = xr.open_zarr("https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1", decode_times=false)

yax_dataset = YAXArrays.open_dataset(python_dataset)

smallsampledata = yax_dataset.analysed_sst[Ti=Near(DateTime(2010,1,1,0)),lon=0..35,lat=40..65]

#Read some data
smallsampledata[:,:].data
````
