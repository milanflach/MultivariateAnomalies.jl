#import MultivariateStats
import Combinatorics


"""
    sMSC(datacube, cycle_length)

subtract the median seasonal cycle from the datacube given the length of year `cycle_length`.

# Examples
```
julia> dc = hcat(rand(193) + 2* sin.(0:pi/24:8*pi), rand(193) + 2* sin.(0:pi/24:8*pi))
julia> sMSC_dc = sMSC(dc, 48)
```
"""
function sMSC(datacube, cycle_length)
    x_out = similar(datacube)
    removeMSC!(datacube, x_out, cycle_length, 1)
    return(x_out)
end

function removeMSC!(xin::AbstractArray{Tp,ndim},xout::AbstractArray{Tp,ndim},NpY::Integer,itimedim::Integer;imscstart::Int=1) where {Tp,ndim}
   # Determine length of reshape dimensions
   s=size(xin)
   l1=itimedim==1 ? 1 : prod(s[1:(itimedim-1)])
   l2=itimedim==ndim ? 1 : prod(s[(itimedim+1):end])
   ltime=s[itimedim]
   #Reshape the cube to squeeze unimportant variables
   xin2=reshape(xin,l1,ltime,l2);
   xout2=reshape(xout,l1,ltime,l2);
   msc=zeros(Float64,NpY); # This is the array where the temp msc is stored
   nmsc=zeros(Int,NpY); # This is for counting how many values were added
   #Start loop through all other variables
   for i2=1:l2,i1=1:l1
       imsc=imscstart
       fill!(msc,zero(Float64))
       fill!(nmsc,zero(Int))
       for itime=1:ltime
           curval=xin2[i1,itime,i2]
           if !isnan(curval)
               msc[imsc]  += curval
               nmsc[imsc] += 1
           end
           imsc =imsc==NpY ? 1 : imsc+1 # Increase msc time step counter
       end
       for i in 1:NpY msc[i] = nmsc[i] > 0 ? msc[i]/nmsc[i] : NaN end # Get MSC by dividing by number of points
       imsc=imscstart
       for i in 1:ltime
           xout2[i1,i,i2] = xin2[i1,i,i2]-msc[imsc]
           imsc =imsc==NpY ? 1 : imsc+1 # Increase msc time step counter
       end
   end
   #Copy data if necessary
   pointer(xout2) != pointer(xout) && copyto!(xout,xout2)
  return(xout)
end


#small helper
function func(i,x)
        return(x[i])
end


"""
    TDE(datacube::Array{tp, 4}, ΔT::Integer, DIM::Int = 3) where {tp}
    TDE(datacube::Array{tp, 3}, ΔT::Integer, DIM::Int = 3) where {tp}

returns an embedded datacube by concatenating lagged versions of the 2-, 3- or 4-dimensional datacube with `ΔT` time steps in the past up to dimension `DIM` (presetting: `DIM = 3`)

# Examples
```
julia> dc = randn(50,3)
julia> TDE(dc, 3, 2)
```
"""
function TDE(datacube::AbstractArray{tp, 4}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(tp, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3), size(datacube, 4) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,:,((dim-1)*size(datacube, 4)+1):(dim*size(datacube, 4))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:,:]
    end
    return(embedded_datacube)
end

function TDE(datacube::AbstractArray{tp, 3}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(tp, (size(datacube, 1) - start +1, size(datacube, 2), size(datacube, 3) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,:,((dim-1)*size(datacube, 3)+1):(dim*size(datacube, 3))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:,:]
    end
    return(embedded_datacube)
end

function TDE(datacube::AbstractArray{tp, 2}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(tp, (size(datacube, 1) - start +1,  size(datacube, 2) * DIM))
    for dim = 1:DIM
    embedded_datacube[:,((dim-1)*size(datacube, 2)+1):(dim*size(datacube, 2))] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT)),:]
    end
    return(embedded_datacube)
end

function TDE(datacube::AbstractArray{tp, 1}, ΔT::Integer, DIM::Int = 3) where {tp}
    start = ((DIM-1)*ΔT+1)
    embedded_datacube = zeros(tp, (size(datacube, 1) - start +1,   DIM))
    for dim = 1:DIM
    embedded_datacube[:,((dim-1)+1):(dim)] =
        datacube[(start - ((dim-1) * ΔT)) : (size(datacube, 1) - ((dim-1) * ΔT))]
    end
    return(embedded_datacube)
end

"""
    mw_VAR(datacube::Array{tp,N}, windowsize::Int = 10) where {tp,N}

compute the variance in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts N dimensional datacubes.

# Examples
```
julia> dc = randn(50,3,3,3)
julia> mw_VAR(dc, 15)
```
"""
function mw_VAR(datacube::AbstractArray{<:AbstractFloat,N}, windowsize::Int = 10) where {N}
  out = zeros(eltype(datacube), size(datacube))
  for k = 1:length(out) out[k] = NaN end
  datacube0mean = datacube .- mean(datacube, dims = 1)
  mw_VAR!(out, datacube0mean, windowsize)
  return(out)
end

"""
    mw_VAR!(out::Array{tp, N}, datacube0mean::Array{tp,N}, windowsize::Int = 10) where {tp,N}

mutating version for `mw_VAR()`. The mean of the input data `datacube0mean` has to be 0. Initialize out properly: `out = datacube0mean` leads to wrong results.

"""
function mw_VAR!(out::AbstractArray{tp, N}, datacube0mean::AbstractArray{tp,N}, windowsize::Int = 10) where {tp, N}
  T = size(datacube0mean, 1)
  for beg = 1:T:length(datacube0mean)
    inner_mw_VAR!(out, datacube0mean, windowsize, beg, T)
  end
   broadcast!(/, out, out, windowsize)
  return(out)
end

"""
    inner_mw_VAR!(out::Array{tp, N}, datacube0mean::Array{tp,N}, windowsize::Int = 10) where {tp,N}

internal function for mw_VAR!()

"""
function inner_mw_VAR!(out, datacube, windowsize, beg, T) # mean already subtracted
  out_beg = floor(Int, windowsize * 0.5) + beg
  for notavailable =  (beg):(out_beg-1)
    out[notavailable] = NaN
  end
  # init x
  x = sum(unsafe_wrap(Array, pointer(datacube, beg), windowsize).^2)
  out[out_beg] = x
  for i = 1:(T-windowsize)
    x = x - datacube[beg + i - 1]^2 + datacube[beg + i - 1 + windowsize]^2
    out[out_beg + i] = x
  end
  for notavailable = (out_beg + (T-windowsize +1) + 1):(beg+T-1)
    out[notavailable] = NaN
  end
  return(out)
end

"""
    mw_AVG(datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}

compute the average in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts N dimensional datacubes.

# Examples
```
julia> dc = randn(50,3,3,3)
julia> mw_AVG(dc, 15)
```
"""
function mw_AVG(datacube::AbstractArray{<:AbstractFloat,N}, windowsize::Int = 10) where {N}
  out = zeros(eltype(datacube), size(datacube))
  mw_AVG!(out, datacube, windowsize)
  return(out)
end

"""
    mw_AVG!(out::Array{tp, N}, datacube::Array{tp,N}, windowsize::Int = 10) where {tp,N}

internal and mutating version for `mw_AVG()`.

"""
function mw_AVG!(out::AbstractArray{tp,N}, datacube::AbstractArray{tp,N}, windowsize::Int = 10) where {tp,N}
  for k = 1:length(out) out[k] = NaN end
  T = size(datacube, 1)
  for beg = 1:T:length(datacube)
    inner_mw_AVG!(out, datacube, windowsize, beg, T)
  end
   broadcast!(/, out, out, windowsize)
  return(out)
end

function inner_mw_AVG!(out::AbstractArray{tp, N}, datacube::AbstractArray{tp, N}, windowsize::Int, beg::Int, T::Int) where {tp,N}
  out_beg = floor(Int, windowsize * 0.5) + beg
  for notavailable =  (beg):(out_beg-1)
    out[notavailable] = NaN
  end
  # init x
  x = sum(unsafe_wrap(Array, pointer(datacube, beg), windowsize))
  out[out_beg] = x
  for i = 1:(T - windowsize)
    x = x - datacube[beg + i - 1] + datacube[beg + i - 1 + windowsize]
    out[out_beg + i] = x
  end
  for notavailable = (out_beg + (T-windowsize +1) + 1):(beg+T-1)
    out[notavailable] = NaN
  end
  return(out)
end



"""
    mw_COR(datacube::Array{tp, 4}, windowsize::Int = 10) where {tp}

compute the correlation in a moving window along the first dimension of the datacube (presetting: `windowsize = 10`).
Accepts 4-dimensional datacubes.

"""
function mw_COR(datacube::AbstractArray{tp, 4}, windowsize::Int = 10) where {tp}
  comb = collect(Combinatorics.combinations(1:size(datacube, 4), 2))
  out = zeros(Float64, size(datacube, 1), size(datacube, 2), size(datacube, 3), size(comb, 1))
  x = zeros(Float64, windowsize, 2)
  for icomb = 1:size(comb, 1)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        for t = 1:(size(datacube, 1)-windowsize)
          x = copyto!(x, datacube[t:(t+windowsize-1),lat,lon,comb[icomb]])
          copyto!(view(out, t + round.(Int, windowsize * 0.5),lat,lon,icomb)
                , cor(view(x,:,1), view(x,:,2)))
        end
      end
    end
  end
  return(comb, out)
end

function innerst_ewma!(z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, beg::Int, len::Int, lambda::Float64 = 0.15) where {tp, N}
    z[beg] = dat[beg]*lambda
    one_m_lambda = 1.0-lambda
    for i = (beg+1):(beg+len-1)
            z[i] = dat[i] * lambda + z[i-1] * one_m_lambda
    end
  return(z)
end

"""
    EWMA(dat,  λ)

Compute the exponential weighted moving average (EWMA) with the weighting parameter `λ` between 0 (full weighting) and 1 (no weighting) along the first dimension of `dat`.
Supports N-dimensional Arrays.

Lowry, C. A., & Woodall, W. H. (1992). A Multivariate Exponentially Weighted Moving Average Control Chart. Technometrics, 34, 46–53.

# Examples
```
julia> dc = rand(100,3,2)
julia> ewma_dc = EWMA(dc, 0.1)
```
"""
function EWMA(dat::AbstractArray{tp, N}, lambda::Float64 = 0.15) where {tp, N}
  Z = similar(dat)
  EWMA!(Z, dat, lambda)
  return(Z)
end


"""
    EWMA!(Z, dat,  λ)

use a preallocated output Z. `Z = similar(dat)` or `dat = dat` for overwriting itself.

# Examples
```
julia> dc = rand(100,3,2)
julia> EWMA!(dc, dc, 0.1)
```
"""
function EWMA!(Z::AbstractArray{tp, N}, dat::AbstractArray{tp, N}, lambda::Float64 = 0.15) where {tp, N}
  T = size(dat, 1)
  LENGTH = length(dat)
  for istart = 1:T:LENGTH
    innerst_ewma!(Z, dat, istart, T, lambda)
  end
  return(Z)
end

"""
    init_MedianCycle(dat::Array{tp}, cycle_length::Int = 46)
    init_MedianCycle(temporal_length::Int[, cycle_length::Int = 46])

initialises an init_MC object to be used as input for `get_MedianCycle!()`. Input is either some sample data or the temporal lenght of the expected input vector
and the length of the annual cycle (presetting: `cycle_length = 46`)
"""
function init_MedianCycle(dat::AbstractArray{tp}, cycle_length::Int = 46) where {tp}
  complete_years = Int(floor(size(dat, 1)/cycle_length))
  cycle_dat = zeros(tp, cycle_length, complete_years)
  cycle_medians = zeros(tp, cycle_length)
  init_MC = (cycle_length, complete_years, cycle_medians, cycle_dat)
  return(init_MC)
end

function init_MedianCycle(temporal_length::Int, cycle_length::Int = 46)
  complete_years = Int(floor(temporal_length/cycle_length))
  cycle_dat = zeros(tp, cycle_length, complete_years)
  cycle_medians = zeros(tp, cycle_length)
  init_MC = (cycle_length, complete_years, cycle_medians, cycle_dat)
  return(init_MC)
end

"""
    get_MedianCycle!(init_MC, dat::Array{tp,1})

Memory efficient version of `get_MedianCycle()`, returning the median cycle in `init_MC[3]`. The `init_MC` object should be created with `init_MedianCycle`.
Can deal with some NaN values.

# Examples
```
julia> using MultivariateAnomalies
julia> dat = rand(193) + 2* sin(0:pi/24:8*pi)
julia> dat[100] = NaN
julia> init_MC = init_MedianCycle(dat, 48)
julia> get_MedianCycle!(init_MC, dat)
julia> init_MC[3]
```
"""
function get_MedianCycle!(init_MC::Tuple{Int64,Int64,Array{tp,1},Array{tp,2}}, dat::Array{tp,1}) where {tp}
  copyto!(init_MC[4], view(dat, 1:(init_MC[2] * init_MC[1])))
  for i = 1:init_MC[1]
    if(all(.!isnan.(view(init_MC[4], i, :))))
      copyto!(view(init_MC[3], i), median(view(init_MC[4], i, :)))
    elseif(all(isnan.(view(init_MC[4], i, :))))
        copyto!(view(init_MC[3], i), NaN)
    else
      copyto!(view(init_MC[3], i), median(view(init_MC[4], i, squeeze(!isnan.(view(init_MC[4], i,:)), 1))))
    end
  end
  return(init_MC[3])
end

"""
    get_MedianCycle(dat::Array{tp,1}, cycle_length::Int = 46)

returns the median annual cycle of a one dimensional data array, given the length of the annual cycle (presetting: `cycle_length = 46`).
Can deal with some NaN values.

# Examples
```
julia> using MultivariateAnomalies
julia> dat = randn(90) + x = sind.(0:8:719)
julia> cycles = get_MedianCycle(dat, 48)
```
"""
function get_MedianCycle(dat::AbstractArray{tp,1}, cycle_length::Int = 46) where {tp}
  init_MC = init_MedianCycle(dat, cycle_length)
  get_MedianCycle!(init_MC, dat)
  return(init_MC[3])
end


"""
    get_MedianCycles(datacube, cycle_length::Int = 46)

returns the median annual cycle of a datacube, given the length of the annual cycle (presetting: `cycle_length = 46`).
The datacube can be 2, 3, 4-dimensional, time is stored along the first dimension.

# Examples
```
julia> using MultivariateAnomalies
julia> dc = hcat(rand(193) + 2* sin(0:pi/24:8*pi), rand(193) + 2* sin(0:pi/24:8*pi))
julia> cycles = get_MedianCycles(dc, 48)
```
"""
function get_MedianCycles(datacube::AbstractArray{tp,4}, cycle_length::Int = 46) where {tp}
dat = zeros(Float64, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(Float64, cycle_length, size(datacube, 2), size(datacube, 3), size(datacube, 4));
# loop
  for var = 1:size(datacube, 4)
    for lon = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :,lat,lon,var))
        copyto!(view(med_cycles_out, :, lat, lon, var), get_MedianCycle!(init_MC, dat))
      end
    end
  end
  return(med_cycles_out)
end

function get_MedianCycles(datacube::AbstractArray{tp,3}, cycle_length::Int = 46) where {tp}
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2), size(datacube, 3));
# loop
  for var = 1:size(datacube, 3)
      for lat = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :, lat,var))
        copyto!(view(med_cycles_out, :, lat, var), get_MedianCycle!(init_MC, dat))
      end
  end
  return(med_cycles_out)
end


function get_MedianCycles(datacube::AbstractArray{tp,2}, cycle_length::Int = 46) where {tp}
dat = zeros(tp, size(datacube, 1));
init_MC = init_MedianCycle(dat, cycle_length);
med_cycles_out = zeros(tp, cycle_length, size(datacube, 2));
# loop
  for var = 1:size(datacube, 2)
        copyto!(dat, view(datacube, :, var))
        copyto!(view(med_cycles_out, :, var), get_MedianCycle!(init_MC, dat))
  end
  return(med_cycles_out)
end

# get moving window loop with MWobj
mutable struct MWobj
    ObsPerYear::Int
    windowsize::Int
    edgecut::Int
    startidx::Int
    numoutvars::Int
    minwindow::Int
    maxwindow::Int
    count::Array{Int, 1}
    iterator_windowcenter::StepRange{Int,Int}
    mw_indat
    mw_outdat
    mw_idx::Array{Int, 1}
    xin
    xout
end


function init_MovingWindow(xin::AbstractArray{tp, 2}; ObsPerYear::Int = 46, windowsize::Int = 11, edgecut::Int = 0, startidx::Int = 1, numoutvars::Int = 0) where {tp}
    if 2*edgecut >= windowsize error("2 * edgecut has to be smaller windowsize, but is $edgecut, windowsize = $windowsize") end
    if !isodd(windowsize) error("windowsize has to be odd, but is $windowsize") end
    if round(Int, size(xin, 1) / ObsPerYear) * ObsPerYear != size(xin, 1) error("ObsPerYear multiplied by some integer is not matching size(xin, 1)") end
    if numoutvars == 0 numoutvars = size(xin, 2) end
    if numoutvars > 1
        mwobj = MWobj(
        ObsPerYear
        ,windowsize
        ,edgecut
        ,startidx
        ,numoutvars
        ,-floor(Int, windowsize / 2.0) # minwindow
        ,floor(Int, windowsize / 2.0) # maxwindow
        , [0] # count
        ,startidx:(windowsize-2*edgecut):ObsPerYear #iterator_windowcenter
        ,zeros(eltype(xin), windowsize * floor(Int, size(xin, 1) / ObsPerYear), size(xin, 2)) #mw_indat
        ,zeros(eltype(xin), windowsize * floor(Int, size(xin, 1) / ObsPerYear), numoutvars) #mw_outdat
        ,zeros(Int, windowsize * floor(Int, size(xin, 1) / ObsPerYear)) # mw_idx
        ,xin
        ,zeros(eltype(xin), size(xin, 1), numoutvars) #xout
        )
    else
        mwobj = MWobj(
        ObsPerYear
        ,windowsize
        ,edgecut
        ,startidx
        ,numoutvars
        ,-floor(Int, windowsize / 2.0) # minwindow
        ,floor(Int, windowsize / 2.0) # maxwindow
        , [0] # count
        ,startidx:(windowsize-2*edgecut):ObsPerYear #iterator_windowcenter
        ,zeros(eltype(xin), windowsize * floor(Int, size(xin, 1) / ObsPerYear), size(xin, 2)) #mw_indat
        ,zeros(eltype(xin), windowsize * floor(Int, size(xin, 1) / ObsPerYear)) #mw_outdat
        ,zeros(Int, windowsize * floor(Int, size(xin, 1) / ObsPerYear)) # mw_idx
        ,xin
        ,zeros(eltype(xin), size(xin, 1), numoutvars) #xout
        )
    end
    return mwobj
end


function getMWData!(mwobj::MWobj, windowcenter::Int)
    xin = mwobj.xin
    mwdata = mwobj.mw_indat
    mwidx = mwobj.mw_idx
    ObsPerYear = mwobj.ObsPerYear
    windowsize = mwobj.windowsize
    minwindow =  mwobj.minwindow
    maxwindow = mwobj.maxwindow
    count = mwobj.count

    count[1] = 0
    for i = windowcenter:ObsPerYear:(size(xin, 1)+windowsize)
      for j = minwindow:maxwindow
        tempidx = i+j
        if tempidx > 0 && tempidx <= size(xin, 1)
          count[1] += 1
          mwidx[count[1]] = tempidx
          for varidx = 1:size(xin, 2)
            mwdata[count[1], varidx] = xin[tempidx, varidx]
          end
        end
      end
    end

  return view(mwobj.mw_indat,1:mwobj.count[1],:)
end

function pushMWResultsBack!(mwobj::MWobj, windowcenter::Int)
    mwdata = mwobj.mw_outdat
    mwidx = mwobj.mw_idx
    xout = mwobj.xout
    ObsPerYear=mwobj.ObsPerYear
    windowsize = mwobj.windowsize
    edgecut = mwobj.edgecut
    minwindow = mwobj.minwindow
    maxwindow = mwobj.maxwindow
    count = mwobj.count

  # assumes Time * Vars
    count[1] = 0
    for i = windowcenter:ObsPerYear:(size(xout, 1)+windowsize)
      for j = minwindow:maxwindow
        tempidx = i+j
        if tempidx > 0 && tempidx <= size(xout, 1)
          count[1] += 1
          if mwidx[count[1]] != tempidx error("not matching mwidx[count] = $(mwidx[count]), check windowsize") end
          if j >= (minwindow + edgecut) && j <= (maxwindow - edgecut)
              for varidx = 1:size(xout, 2)
                xout[tempidx, varidx] = mwdata[count[1], varidx]
              end
          end
        end
      end
    end

    return view(mwobj.mw_outdat, 1:mwobj.count[1], :)
end


"""
    mapMovingWindow(function2mw, x, args...; ObsPerYear::Int = 46, windowsize::Int = 9, edgecut::Int = 0, startidx::Int = 1, numoutvars::Int = 0)
    mapMovingWindow(function2mw, x; ...)
apply a function (function2mw) in a moving window encompassing all years and running along the time, e.g. apply the function to all summers, then summers + 1 timestep ...
results are written to the center of the respective windowsize. Input axes are time or time-variables. The number of output variables can be different from the number of input variables, specified in numoutvars.
e.g. transforming the variables in normalised ranks between zero and one to get rid of heteroscedasticity would look like:
    using MultivariateAnomalies
    x = randn(10*46, 3)
    mapMovingWindow(get_quantile_scores, x, numoutvars = size(x, 2))
"""
function mapMovingWindow(function2mw, x, args...; ObsPerYear::Int = 46, windowsize::Int = 9, edgecut::Int = 0, startidx::Int = 1, numoutvars::Int = 0)
  mwobj = init_MovingWindow(x, ObsPerYear = ObsPerYear, windowsize = windowsize, edgecut = edgecut, numoutvars = numoutvars, startidx = startidx)
  if numoutvars > 1
      for windowcenter = mwobj.iterator_windowcenter
        getMWData!(mwobj, windowcenter)
        # do something with mwobj.mw_indat and write the results to mwobj.mw_outdat
        xout = view(mwobj.mw_outdat, 1:mwobj.count[1], :)
        xin = view(mwobj.mw_indat, 1:mwobj.count[1], :)
        xout[:] = function2mw(xin, args...)
        pushMWResultsBack!(mwobj, windowcenter)
    end
  else
      for windowcenter = mwobj.iterator_windowcenter
        getMWData!(mwobj, windowcenter)
        # do something with mwobj.mw_indat and write the results to mwobj.mw_outdat
        xout = view(mwobj.mw_outdat, 1:mwobj.count[1])
        xin = view(mwobj.mw_indat, 1:mwobj.count[1], :)
        xout[:] = function2mw(xin, args...)
        pushMWResultsBack!(mwobj, windowcenter)
    end
  end
  return mwobj.xout
end

function mapMovingWindow(function2mw, x; ObsPerYear::Int = 46, windowsize::Int = 9, edgecut::Int = 0, startidx::Int = 1, numoutvars::Int = 0)
  mwobj = init_MovingWindow(x, ObsPerYear = ObsPerYear, windowsize = windowsize, edgecut = edgecut, numoutvars = numoutvars, startidx = startidx)
  if numoutvars > 1
      for windowcenter = mwobj.iterator_windowcenter
        getMWData!(mwobj, windowcenter)
        # do something with mwobj.mw_indat and write the results to mwobj.mw_outdat
        xout = view(mwobj.mw_outdat, 1:mwobj.count[1], :)
        xin = view(mwobj.mw_indat, 1:mwobj.count[1], :)
        xout[:] = function2mw(xin)
        pushMWResultsBack!(mwobj, windowcenter)
    end
  else
      for windowcenter = mwobj.iterator_windowcenter
        getMWData!(mwobj, windowcenter)
        # do something with mwobj.mw_indat and write the results to mwobj.mw_outdat
        xout = view(mwobj.mw_outdat, 1:mwobj.count[1])
        xin = view(mwobj.mw_indat, 1:mwobj.count[1], :)
        xout[:] = function2mw(xin)
        pushMWResultsBack!(mwobj, windowcenter)
    end
  end
  return mwobj.xout
end

###################################
#end
