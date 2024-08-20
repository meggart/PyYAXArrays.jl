module PyYAXArrays

using PythonCall, DiskArrays, YAXArrayBase
import CondaPkg
export XArrayDataset


const xr = Ref{Py}()
function __init__()
    xr[] = pyimport("xarray")
    push!(YAXArrayBase.backendlist,:xarray => XArrayDataset)
    pushfirst!(YAXArrayBase.backendregex,r"^xr::"=>XArrayDataset)
end

# The new type with a field for the Python object being wrapped.
struct XRDataArrayDiskArray{T,N} <: AbstractDiskArray{T,N}
    pyar::Py #Pointer to the DataArray
    eltype::Type{T}
    ndims::Val{N}
end
function XRDataArrayDiskArray(da::Py)
    dtstr = pyconvert(String,da.dtype.descr[0][1])
    T = try
        PythonCall.Wrap.pyarray_typestrdescr_to_type(dtstr,PythonCall.PyNULL)
    catch
        Any
    end
    ndims = length(da.shape)
    XRDataArrayDiskArray(da,T,Val(ndims))
end
Base.size(a::XRDataArrayDiskArray{<:Any,N}) where N = reverse(pyconvert(NTuple{N,Int},a.pyar.shape))
function DiskArrays.eachchunk(a::XRDataArrayDiskArray)
    cs = a.pyar.chunks
    if pynot(cs)  
        s = pyconvert(Tuple,a.pyar.shape)
        return DiskArrays.GridChunks(s,s)
    else 
        diskarraychunks = ntuple(pylen(cs)) do i
            DiskArrays.chunktype_from_chunksizes(pyconvert(Vector,cs[i-1]))
        end
        return DiskArrays.GridChunks(reverse(diskarraychunks))
    end
end

DiskArrays.haschunks(::XRDataArrayDiskArray) = DiskArrays.Chunked()
function DiskArrays.readblock!(a::XRDataArrayDiskArray{<:Any,N}, xout, r::OrdinalRange...) where N
    slices_py = map(r) do ra
        pyslice(first(ra)-1,last(ra),step(ra))
    end
    r = pygetitem(a.pyar,reverse(slices_py)).values
    wrappedar = PythonCall.Wrap.PyArray(r,copy=false,array=true)
    perm = reverse(ntuple(identity,N))
    permutedims!(xout,wrappedar,perm)
end
# Says that the object is a wrapper.
PythonCall.ispy(x::XRDataArrayDiskArray) = true

# Says how to access the underlying Python object.
PythonCall.Py(x::XRDataArrayDiskArray) = x.pyar


struct XArrayDataset
    xrdataset::Py
end
function XArrayDataset(p::AbstractString;mode="r")
    p = replace(p,r"^xr::"=>"")
    XArrayDataset(xr[].open_zarr(p))
end
Base.haskey(a::XArrayDataset,k) = in(k,pylist(a.xrdataset.data_vars)) || in(k,pylist(a.xrdataset.coords))

YAXArrayBase.get_var_handle(ds::XArrayDataset, name) = XRDataArrayDiskArray(pygetattr(ds.xrdataset,name))

YAXArrayBase.to_dataset(py::Py; driver=:xarray) = XArrayDataset(py)

function YAXArrayBase.get_varnames(ds::XArrayDataset)
    vars = pylist(ds.xrdataset.data_vars)
    coords = pylist(ds.xrdataset.coords)
    [pyconvert(Vector,vars);pyconvert(Vector,coords)]
end


function YAXArrayBase.get_var_dims(ds::XArrayDataset, name)
    v = pygetattr(ds.xrdataset,name)
    reverse!(pyconvert(Vector,v.dims))
end

function YAXArrayBase.get_var_attrs(ds::XArrayDataset, name)
    v = pygetattr(ds.xrdataset,name)
    pyconvert(Dict{String,Any},v.attrs)
end

function YAXArrayBase.get_global_attrs(ds::XArrayDataset)
    pyconvert(Dict{String,Any},ds.xrdataset.attrs)
end

end
